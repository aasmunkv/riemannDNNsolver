"""
mlp2d.network
=============
TODO: Write this doc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

in_jupy = False
try:
    cfg = get_ipython().__class__.__name__
    if cfg == 'ZMQInteractiveShell':
        in_jupy = True
except NameError:
    in_jupy = False

if in_jupy:
    from tqdm import tqdm_notebook as tqdm # for notebook
else:
    from tqdm import tqdm # for terminal


class Cell(nn.Module):
    def __init__(self, dimensions, activation=F.relu, final_activation=False):
        super(Cell, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu" # then use obj.to(self.device) to cast to GPU/CPU memory

        self.dimensions = dimensions
        self.size = len(dimensions)
        self.activation = activation
        self.final_activation = final_activation

        self.layer_inp = nn.Linear(dimensions[0], dimensions[1])
        self.layer_hid = nn.ModuleList([
            nn.Linear(dimensions[i], dimensions[i+1]) for i in range(1, self.size - 2)
        ])
        self.layer_out = nn.Linear(dimensions[-2], dimensions[-1])
            
    def forward(self, inp):
        inp = inp.type(torch.float32).to(self.device)
        out = self.layer_inp(inp)
        out = self.activation(out)
        for layer in self.layer_hid:
            out = self.activation(layer(out).to(self.device))
        out = self.layer_out(out)
        return out if not self.final_activation else self.activation(out)


class Network(nn.Module):
    def __init__(self, 
                N, dimensions,
                activation=torch.relu,
                final_activation=False,
                optimizer=torch.optim.Adam):
        super().__init__()

        # seed to obtain consistent results
        torch.random.manual_seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        self.device = "cuda" if torch.cuda.is_available() else "cpu" # then use obj.to(self.device) to cast to GPU memory

        self.N = N 
        self.dimensions = dimensions

        self.trained_epochs = 0
        self.epochs = None
        self.batchsize = None
        
        self.activation = activation 
        self.final_activation = final_activation

        self.network = Cell(dimensions, activation=activation).to(self.device)

        self.opt = optimizer(self.network.parameters())
        self.loss_train = None
        self.loss_val = None
        self.weights = None

    def backward(self, data_train, data_val, epochs, batchsize, destination, name):
        self.trained_epochs += epochs
        self.epochs, self.batchsize = epochs, batchsize
        inp_train = data_train[:,:self.dimensions[0]]
        out_train = data_train.data[:,self.dimensions[0]].to(self.device)

        inp_val = data_val[:,:self.dimensions[0]]
        out_val = data_val.data[:,self.dimensions[0]].to(self.device)

        # Make a 'data-feeder'
        sampler = torch.utils.data.DataLoader(
            range(self.N), 
            batch_size=self.batchsize,
            shuffle=True
        )
        # Create variables for saving history
        if self.loss_train is None and self.loss_val is None and self.weights is None:
            self.loss_train = []
            self.loss_val = []
            self.weights = []
            val_loss = self.network.forward(inp_val).squeeze().to(self.device)
            val_loss = nn.MSELoss()(val_loss, out_val.type(torch.float32))
            self.loss_val.append(val_loss)
        else:
            self.loss_train = list(self.loss_train)
            self.loss_val = list(self.loss_val)
            self.weights = list(self.weights)
        best_loss = np.inf
        cur_loss = np.inf
        pbar = tqdm(
            total=self.epochs, 
            desc='Training progress (loss:       )', 
            bar_format = '{desc}: {percentage:3.0f}%{bar}Epoch: {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        for epoch in range(self.epochs):
            if self.device=='cuda': torch.cuda.empty_cache()
            loss_tmp = []
            for i, batch in enumerate(sampler):
                self.opt.zero_grad()
                loss_out = self.network.forward(inp_train[batch]).to(self.device).squeeze()
                loss_tar = out_train[batch].to(self.device)
                cur_loss = nn.MSELoss()(loss_out.to(self.device), loss_tar.type(torch.float32).to(self.device))
                del loss_out, loss_tar
                cur_loss.backward(retain_graph=True)
                self.opt.step()
                self.loss_train.append(cur_loss.data)
                # Save model if it is better than the previous best
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    destination = '.' if (destination == '') else destination
                    torch.save(self.network, destination+'/'+'model_'+name+'.pt')
                del cur_loss
            # Save validation loss and average of training loss
            val_loss = self.network.forward(inp_val).squeeze().to(self.device)
            val_loss = nn.MSELoss()(val_loss, out_val.type(torch.float32))
            self.loss_val.append(val_loss)
            weight_beg = [self.network.layer_inp.weight.cpu().detach().numpy()]
            weight_mid = [l.weight.cpu().detach().numpy() for l in self.network.layer_hid]
            weight_end = [self.network.layer_out.weight.cpu().detach().numpy()]
            self.weights.append(weight_beg + weight_mid + weight_end)
            pbar.set_description(desc='Training progress (best loss: {0:.2e})'.format(best_loss))
            pbar.update(1)
        pbar.close()
        self.loss_train = torch.tensor(self.loss_train)
        self.loss_val = torch.tensor(self.loss_val)

    def backward_newLoss(self, data_train, data_val, epochs, batchsize, destination, name):
        """
        NB: Not in use due to the fact that it must be created training data for this
        specific use, as training data for 1D new loss is created. This would most likely
        take forevery if we want to create sufficient amount of data...
        """
        def loss_var(inp, out):
            """
            Calculates the loss wrt. the numeric solution to conservation law.
            Inputs:
                inp - input of network
                out - output of network
            Output:
                loss - out-value to send into actual loss function
            """
            inp = inp.to(self.device)
            out = out.to(self.device)
            out = out.squeeze(-1)
            loss = torch.zeros_like(inp, requires_grad=False)

            dx = 1/self.N
            dt = dx/(torch.max(torch.abs(self.dfdu(inp)))) # TODO: Why correct without multiplying with Courant coefficient?!
            C = dt/dx

            loss[:,:-1] = inp[:,:-1] - C*(out[:,1:] - out[:,:-1])
            loss[:,-1] = inp[:,-1] - C*(out[:,0] - out[:,-1])
            return loss
        # set index tensor
        N_ind = torch.arange(0,self.N).reshape((self.N,1))
        inp_ind = torch.cat((N_ind.roll(1),N_ind),1) # call: data[inp_ind]
        # set epochs and batchsize
        self.trained_epochs += epochs
        self.epochs, self.batchsize = epochs, batchsize
        # extract data from input
        inp_train = data_train[:,:self.dimensions[0]]
        out_train = data_train.data[:,self.dimensions[0]].to(self.device)
        inp_val = data_val[:,:self.dimensions[0]]
        out_val = data_val.data[:,self.dimensions[0]].to(self.device)

        # Make a 'data-feeder'
        sampler = torch.utils.data.DataLoader(
            range(self.N), 
            batch_size=self.batchsize,
            shuffle=True
        )
        # Create variables for saving history
        if self.loss_train is None and self.loss_val is None and self.weights is None:
            self.loss_train = []
            self.loss_val = []
            self.weights = []
            val_loss = (self.network.forward(inp_val[:,inp_ind].to(self.device)).data)  #[:,inp_ind] of inp?
            val_loss = loss_var(inp_val, val_loss)
            val_loss = nn.L1Loss()(val_loss, out_val.to(self.device))
            self.loss_val.append(val_loss)
            #val_loss = self.network.forward(inp_val).squeeze().to(self.device)
            #val_loss = nn.MSELoss()(val_loss, out_val.type(torch.float32))
            #self.loss_val.append(val_loss)
        else:
            self.loss_train = list(self.loss_train)
            self.loss_val = list(self.loss_val)
            self.weights = list(self.weights)
        best_loss = np.inf
        cur_loss = np.inf
        pbar = tqdm(
            total=self.epochs, 
            desc='Training progress (loss:       )', 
            bar_format = '{desc}: {percentage:3.0f}%{bar}Epoch: {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        for epoch in range(self.epochs):
            torch.cuda.empty_cache()
            loss_tmp = []
            for i, batch in enumerate(sampler):
                self.opt.zero_grad()
                loss_out = self.network.forward(inp_train[batch][:,inp_ind]).to(self.device).squeeze()
                loss_out = loss_var(inp_train[batch], loss_out)
                loss_tar = out_train[batch].to(self.device)
                cur_loss = nn.L1Loss()(loss_out, loss_tar)
                #cur_loss = nn.MSELoss()(loss_out.to(self.device), loss_tar.type(torch.float32).to(self.device))
                del loss_out, loss_tar
                cur_loss.backward(retain_graph=True)
                self.opt.step()
                self.loss_train.append(cur_loss.data)
                # Save model if it is better than the previous best
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    destination = '.' if (destination == '') else destination
                    torch.save(self.network, destination+'/'+'model_'+name+'.pt')
                del cur_loss
            # Save validation loss and average of training loss
            #val_loss = self.network.forward(inp_val).squeeze().to(self.device)
            val_loss = (self.network.forward(inp_val[:,inp_ind].to(self.device)).data)
            #val_loss = nn.MSELoss()(val_loss, out_val.type(torch.float32))
            val_loss = loss_var(inp_val, val_loss)
            val_loss = nn.L1Loss()(val_loss, out_val.to(self.device))
            self.loss_val.append(val_loss)
            weight_beg = [self.network.layer_inp.weight.cpu().detach().numpy()]
            weight_mid = [l.weight.cpu().detach().numpy() for l in self.network.layer_hid]
            weight_end = [self.network.layer_out.weight.cpu().detach().numpy()]
            self.weights.append(weight_beg + weight_mid + weight_end)
            pbar.set_description(desc='Training progress (best loss: {0:.2e})'.format(best_loss))
            pbar.update(1)
        pbar.close()
        self.loss_train = torch.tensor(self.loss_train)
        self.loss_val = torch.tensor(self.loss_val)

    @property
    def history(self):
        hist = []
        if self.loss_train is not None:
            hist.append(self.loss_train)
        if self.loss_val is not None:
            hist.append(self.loss_val)
        return hist

    @property
    def history_weight(self):
        hist = []
        if self.weights is None:
            return (hist)
        return self.weights