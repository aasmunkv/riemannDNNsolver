"""
datagenerator.data1d_newLoss
============================

Description
-----------
Provides data for training of mlp1d.network using backward_newLoss.
This is therefore data to test a new type of loss function of 1D Riemann problems.

Usage
-----
Constructor (Dataset)
    __init__(self, M, N, K, f, dfdu)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

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

class Dataset:
    def __init__(self, M, N, K, f, dfdu):
        """
            Create m x n data points.
            Inputs:
                M - number of rows of data.
                    type: int
                N - mesh size of the two arrays in column direction.
                    type: int
                K - constant to use for variation when creating Fourier coefficients.
                    type: int
                f - flux function of conservation law.
                    type: callable
                dfdu - the derivative of flux function.
                    type: callable
        """
        super().__init__()

        # seed to obtain consistent results
        torch.random.manual_seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        # check types of input
        if not callable(f) or not callable(dfdu):
            raise TypeError('f, dfdu need to be callable functions.')
        if not isinstance(M, int) or not isinstance(N, int) or not isinstance(K, int):
            raise TypeError("Inputs 'M', 'N' and 'K' need to be integers.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.M = M
        self.N = N
        self.K = K
        self.f = f
        self.dfdu = dfdu

        self.data = torch.zeros((self.M, 2, self.N))

    @property
    def create(self):
        """
            Create Mx2xN data points using K to set coefficients of Fourier sine series.
            This yields input data (data[i,0,:]) and is the base for output (data[i,1,:]).
        """
        dx = 1/self.N
        x = torch.transpose( torch.linspace(dx, 1, self.N).expand(1, self.N), 0, 1 ).to(self.device)
        pbar = tqdm(
            total=self.M, 
            desc='Creating progress', 
            bar_format = '{desc}: {percentage:3.0f}%{bar}Column: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        for i in range(self.M):
            coeffs = torch.tensor([
                np.random.normal(0, 1/k) for k in range(1, self.K+1)    #Can also use np.random.uniform(-1/k, 1/k), not tested which is best
            ]).reshape(self.K,1).to(self.device)

            k_inds = torch.linspace(1, self.K, self.K).to(self.device)

            v_inp = (torch.sin(x*k_inds*np.pi) @ coeffs).squeeze()

            del k_inds, coeffs

            dt = dx/(2*torch.max(torch.abs(self.dfdu(v_inp)))) # CFL coeff = 0.5
            
            # A right shift of v_inp (serves as right input of godunov flux):
            v_inp_neg = v_inp.roll(1)
            # A left shift of v_inp (serves as left input of godunov flux):
            v_inp_pos = v_inp.roll(-1)
                    
            F_pos = self.godunovFlux(v_inp, v_inp_pos)
            F_neg = self.godunovFlux(v_inp_neg, v_inp)
            
            v_out = v_inp - (dt/dx)*(F_pos - F_neg)
            
            self.data[i,0,:] = v_inp[:]
            self.data[i,1,:] = v_out[:]
            pbar.update(1)
        pbar.close()

    def godunovFlux(self, u_l, u_r):
        """
            Calculates the Godunov flux when creating data.
            Input:
                u_l - left cell values
                    type: tensor
                u_r - right cell values 
                    type: tensor
        """
        if (type(u_l) is not torch.Tensor):
            raise TypeError("Not tensorinput!")

        u_l = u_l.type(torch.float64).to(self.device)
        u_r = u_r.type(torch.float64).to(self.device)

        arr = torch.linspace(0,1,1000, dtype=torch.float64).to(self.device)
        
        arr_mesh, u_l_mesh = torch.meshgrid(arr, u_l)
        _, u_r_mesh = torch.meshgrid(arr, u_r)
        del arr

        diff = u_r_mesh-u_l_mesh
        del u_r_mesh, u_l_mesh
        flip = (diff < 0)*(-1)

        arr_mesh_flip = torch.abs(arr_mesh + flip)

        u_min = torch.min(u_l, u_r).reshape(u_l.size(),1)
        
        arr_scaled = torch.add(arr_mesh_flip*torch.abs(diff), u_min)
        del diff, arr_mesh_flip

        arr_f = self.f(arr_scaled)

        arr_f_min = torch.min(arr_f, axis=0)[0]
        arr_f_max = torch.max(arr_f, axis=0)[0]
        del arr_f

        cond = torch.stack(( (u_l < u_r), (u_r <= u_l) ))
        del u_l, u_r

        term1 = arr_f_max*cond[1]
        term2 = arr_f_min*cond[0]
        term_add = term1+term2
        
        return term_add

    def save(self, destination, filename):
        """
            Saves 'self.data' at given destination, with given filename.
            Input:
                destination - name of folder to save data in, must be existing
                    type: string
                filename - name of file to be saved, without postfix, must be non-existing
                    type: string
            Output: None
        """
        if destination == '':
            destination = '.'

        if self.data is None:
            raise UnboundLocalError("'self.data' is None.")

        if not os.path.exists(destination):
            raise FileNotFoundError("The given directory/folder does not exist.")

        if os.path.exists(destination+'/'+'data_'+filename+'.pt'):
            raise FileExistsError("The filename already exists.")

        torch.save(self.data, destination+'/'+'data_'+filename+'.pt')
        print("Data is saved in "+destination+'/'+'data_'+filename)
    
    def load(self, destination, filename):
        """
            Loads self.data from given filname and destination.
            Input:
                destination - name of folder to fetch data from, must be existing
                    type: string
                filename - name of file to be fetched, without postfix, must existing
                    type: string
            Output: None
        """
        if not os.path.exists(destination):
            raise FileNotFoundError("The directory doies not exist.")

        if not os.path.exists(destination+'/'+'data_'+filename+'.pt'):
            raise FileExistsError("The file does not exist.")

        self.data = torch.load(destination+'/'+'data_'+filename+'.pt')
        print("Data is loaded from "+destination+'/'+'data_'+filename)

        self.data = self.data.cuda()

