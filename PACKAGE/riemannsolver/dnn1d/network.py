"""
mlp1d.network
=============
Create a network for 1 dimensional Riemann problems, essentially 2 inputs, and 1 output.
The number of inputs and outputs are chosen by user, but code is created to work on
networks which have trainingdata of 2 in 1 out. 

Example code
------------
    # set N, dimensions, dfdu
    net = network.Network(
        N=N, 
        dimensions=dimensions,
        dfdu=dfdu
    )
    # set epochs, batchsize, data_train, data_val, destination, name
    net.backward(
        epochs = epochs,
        batchsize = batchsize,
        data_train=data_trn,
        data_val=data_val,
        destination=destination,
        name=name
    )
    # plot training and result using flatplotlib.netplot
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
        """
            Inputs:
                dimensions - number of node in network layers,
                    type: torch.tensor,
                    structure: [dim(input), ..., dim(hidden layers), ..., dim(output)]
                activation - activation layer to use in all layers,
                    default: torch.relu
                final_activation - whether or not to have activation i last layer,
                    type: boolean,
                    default: False
        """
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

        #self.drop = nn.Dropout(p=0.05)
            
    def forward(self, inp):
        """
            Feed forward method of network. 
            Input:
                inp - input tensor to insert into network,
                    type: torch.tensor,
                    size: self.dimension_layers[0]
            Output:
                out - output of network,
                    type: torch.tensor,
                    size: self.dimension_layers[-1]
        """
        inp = inp.to(self.device)
        out = self.layer_inp(inp)
        out = self.activation(out)
        for layer in self.layer_hid:
            out = self.activation(layer(out).to(self.device))
            #out = self.drop(out)
        out = self.layer_out(out)
        return out if not self.final_activation else self.activation(out)


class Network(nn.Module):
    def __init__(self, 
                N, dimensions, dfdu,
                activation=torch.relu,
                final_activation=False,
                optimizer=torch.optim.Adam):
        """
            The outer structure of the network with the new loss function. This function 
            takes dataset of M x 2 x N dimension and uses N data points a basis for each 
            run through of the training process. Then we calculate the loss with respect 
            to all N outputs and back propagate through the network structure called Cell, 
            to correct errors.
            Input:
                N - number of data points to throw at the network each time,
                    type: int
                dimensions - list of dimensions in layers of the inner network structure,
                    type: tuple, list, numpy.array, torch.tensor
                loss - the loss function to use when training the inner network,
                    default: torch.nn.L1Loss
                activation - activation layer to use in all layers,
                    default: torch.relu
                final_activation - whether or not to have activation i last layer,
                    type: boolean,
                    default: False
            Additional:
                cells - list of N objects, for training with N dimensional data.
                    type: Cell
                inp_ind - N x 2 dimensional tensor of cycle indexes for training data.
                    structure: [ [N, 0], [0, 1], ..., [N-1, N] ]
        """
        super().__init__()

        # seed to obtain consistent results
        torch.random.manual_seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        self.device = "cuda" if torch.cuda.is_available() else "cpu" # then use obj.to(self.device) to cast to GPU memory

        self.dfdu = dfdu
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
        """
        Inputs:
            data_train - data for training
                type: tensor
            data_val - data for validation
                type: tensor
            epochs - number of epochs to train 
            batchsize - batchsize to train simultaneously
            destination - destination to save model
            name - name to save model as
        """
        self.trained_epochs += epochs
        self.epochs, self.batchsize = epochs, batchsize
        inp_train_l = data_train.data[:,0].to(self.device)
        inp_train_l = inp_train_l.reshape((len(inp_train_l),1))
        inp_train_r = data_train.data[:,1].to(self.device)
        inp_train_r = inp_train_r.reshape((len(inp_train_r),1))
        inp_train = torch.cat((inp_train_l,inp_train_r),dim=1)
        out_train = data_train.data[:,2].to(self.device)

        inp_val_l = data_val.data[:,0].to(self.device)
        inp_val_l = inp_val_l.reshape((len(inp_val_l),1))
        inp_val_r = data_val.data[:,1].to(self.device)
        inp_val_r = inp_val_r.reshape((len(inp_val_r),1))
        inp_val = torch.cat((inp_val_l,inp_val_r),dim=1)
        out_val = data_val.data[:,2].to(self.device)

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
            val_loss = nn.MSELoss()(val_loss, out_val)
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
                cur_loss = nn.MSELoss()(loss_out.to(self.device), loss_tar.to(self.device))
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
            val_loss = nn.MSELoss()(val_loss, out_val)
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
        Inputs:
            data_train - data for training
                type: tensor
            data_val - data for validation
                type: tensor
            epochs - number of epochs to train 
            batchsize - batchsize to train simultaneously
            destination - destination to save model
            name - name to save model as
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
        inp_train, out_train = data_train[:,0,:], data_train[:,1,:]
        inp_val, out_val = data_val[:,0,:], data_val[:,1,:]
        # Make a 'data-feeder'
        sampler = torch.utils.data.DataLoader(
            range(inp_train.shape[0]), 
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
                out = self.network.forward(inp_train[batch][:,inp_ind].to(self.device))
                loss_out = loss_var(inp_train[batch], out)
                loss_tar = out_train[batch].to(self.device)
                cur_loss = nn.L1Loss()(loss_out, loss_tar)
                cur_loss.backward(retain_graph=True)
                self.opt.step()
                self.loss_train.append(cur_loss.data)
                # Save model if it is better than the previous best
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    destination = '.' if (destination == '') else destination
                    torch.save(self.network, destination+'/'+'model_'+name+'.pt')
            # Save validation loss and average of training loss
            val_loss = (self.network.forward(inp_val[:,inp_ind].to(self.device)).data)  #[:,inp_ind] of inp?
            val_loss = loss_var(inp_val, val_loss)
            val_loss = nn.L1Loss()(val_loss, out_val.to(self.device))
            self.loss_val.append(val_loss)
            weight_beg = [self.network.layer_inp.weight.cpu().detach().numpy()]
            weight_mid = [l.weight.cpu().detach().numpy() for l in self.network.layer_hid]
            weight_end = [self.network.layer_out.weight.cpu().detach().numpy()]
            self.weights.append(weight_beg + weight_mid + weight_end)
            pbar.update(1)
            pbar.set_description(desc='Training progress (loss: {0:.2e})'.format(cur_loss))
            # print(f'epoch: {epoch:5d} - loss: {cur_loss.item():9.6e}')
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

