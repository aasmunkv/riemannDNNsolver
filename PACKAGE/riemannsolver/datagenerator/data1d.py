"""
datagenerator.data1d
====================

Description
-----------
Provides data for training of mlp1d.

Usage
-----
Constructor (Dataset)
    __init__(self, N, f, loc=0.0, scale=1.0)

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
    def __init__(self, N, f, loc=0.0, scale=1.0):
        """
        Create m x n data points for normal 1 dimensional conservation law network.
        Inputs:
            N - size of dataset.
                type: int
            f - flux function (of the scalar conservation law)
                type: callable function
            loc - mean of normalized data
                type: float
            scale - standard deviation of normalized data
                type: float
        Additional:
            data - generated data
        """
        super().__init__()

        # seed to obtain consistent results
        torch.random.manual_seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        

        if not callable(f):
            raise TypeError('f, dfdu are %s, %s. Both need to be callable.' % (type(f)))
        
        if not isinstance(N, int):
            raise TypeError("Inputs 'M', 'N' and 'K' need to be integers.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.god_flux_mesh_size = 10000
        
        self.N = N
        self.f = f
        self.l = loc 
        self.s = scale
        self.data = torch.zeros((self.N, 3)).to(self.device)

    @property
    def create(self):
        torch.cuda.empty_cache()

        #noise_x = torch.rand(self.N)*2-1
        #noise_y = torch.rand(self.N)*2-1

        self.data[:,0] = torch.randn((self.N)) * self.s + self.l #+ noise_x
        self.data[:,1] = torch.randn((self.N)) * self.s + self.l #+ noise_y
        
        pbar = tqdm(
            total=self.N, 
            desc='Creating progress', 
            bar_format = '{desc}: {percentage:3.0f}%{bar}Column: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        for i in range(0,self.N, 1000):
            U_L = self.data[i:i+1000,0].type(torch.float64).to(self.device)
            U_R = self.data[i:i+1000,1].type(torch.float64).to(self.device)
            # Set up identity arr from 0 to 1 (to be scaled wrt U_L and U_R)
            arr = torch.linspace(0,1,self.god_flux_mesh_size, dtype=torch.float64).to(self.device)
            # Create mesh grids
            arr_mesh, U_L_mesh = torch.meshgrid(arr, U_L)
            _, U_R_mesh = torch.meshgrid(arr, U_R)
            del arr
            # Set the arr_mesh correct by flip if U_L>U_R and then scale it
            diff = (U_R_mesh-U_L_mesh)
            del U_R_mesh,U_L_mesh
            flip = (diff < 0)*(-1)
            arr_mesh_flip = torch.abs(arr_mesh + flip)
            del flip
            U_min = torch.min(U_L, U_R).reshape(U_L.size(),1)
            # Scale arr properly from min(U_L,U_R) to max(U_L,U_R)
            arr_scaled = torch.add(arr_mesh_flip*torch.abs(diff), U_min)
            del diff,arr_mesh_flip
            # Calculate function values
            arr_f = self.f(arr_scaled)
            del arr_scaled
            # Calculate flux
            flux_min = torch.min(arr_f, axis=0)[0]
            flux_max = torch.max(arr_f, axis=0)[0]
            del arr_f
            # Set the conditions
            cond = torch.stack(( (U_L < U_R), (U_R <= U_L) ))
            del U_L, U_R
            cond_shape = cond.shape[1:]
            cond = cond.reshape(cond.size()[0], flux_min.size()[-1])
            # Create return variable
            self.data[i:i+1000,2] = (flux_min*cond[0] + flux_max*cond[1]).reshape(cond_shape)
            del flux_min,flux_max
            pbar.update(1000)
        pbar.close()

    @property
    def get_data(self):
        """
        Action: 
            returns the data if it is not None.
        Inputs: None
        Output:
            self.data
        """
        if self.data is None:
            raise UnboundLocalError("'self.data' is None.")
        return self.data

    def save(self, destination, filename):
        """
        Action:
            Saves data if 'self.data' is not None.
            Includes some tests to avoid overwriting existing datasets.
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
        print("Data is saved in "+destination+'/'+filename)
    
    def load(self, destination, filename):
        """
        Action:
            Loads data from file with given name within directory with given name.
        Input:
            destination - name of folder to fetch data from, must be existing
                type: string
            filename - name of file to be fetched, without postfix, must existing
                type: string
        Output: None
        """
        if not os.path.exists(destination):
            raise FileNotFoundError("The directory doies not exist.")

        if not os.path.exists(destination+'/'+filename+'.pt'):
            raise FileExistsError("The file does not exist.")

        self.data = torch.load(destination+'/'+filename+'.pt')
        print("Data is loaded from "+destination+'/'+filename)