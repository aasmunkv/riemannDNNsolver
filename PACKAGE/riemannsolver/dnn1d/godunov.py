"""
mlp1d.godunov
=============

Description
-----------
Uses a pretrained network as approximation for the Godunov flux. Otherwise, it is used 
regular Godunov method algorithm for 1 dimensional Riemann problems.

Usage
-----
Constructor (Dataset)
    __init__(self, f, dfdu, u0, bnd_cond, xmin, xmax, Nx, network, T=1.0, C=0.5)

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Godunov:
    def __init__(self, f, dfdu, u0, bnd_cond, xmin, xmax, Nx, network, T=1.0, C=0.5):
        """
            Inputs:
                f - flux function
                    type: callable
                dfdu - derivative of flux function
                    type: callable
                u0 - initial function
                    type: callable
                bnd_cond - boundary condition
                    type: string
                xmin - minimum value in x interval
                    type: float
                xmax - maximum value in x interval
                    type: float
                Nx - size of mesh on x interval
                    type: int
                network - pretrained network to use for fetching numerical flux of scheme
                    type: riemannsolver.mlp1d.network.Network
                T - maximum time value
                    type: float
                    default: 1.0
                C - Courant coefficient, choose wisely for stability
                    type: float
                    default: 0.5
        """
        if not callable(f) or not callable(dfdu) or not callable(u0):
            raise TypeError("'f', 'dfdU' and 'U0' need to be functions.")
        if not isinstance(bnd_cond, str):
            raise TypeError("'bnd_cond' needs to be string with boundary condition.")
        
        self.f = lambda U: f(U)
        self.dfdU = lambda U: dfdu(U)
        self.u0 = lambda x: u0(x)
        
        self.bnd_cond = bnd_cond
        
        self.xmin, self.xmax, self.Nx = xmin, xmax, Nx
        self.x = torch.linspace(xmin, xmax, Nx)
        self.dx = (xmax - xmin)/(Nx-1)
        self.T = T
        self.C = C
        self.dt, self.Nt = None, None
        
        self.net = network

        self.u = [self.u0(self.x)]
        
    def set_dt(self):
        """
            Sets the mesh size and step size in temporal dimension.
            Input:
                u - cell average values
            Output: None
        """
        a = torch.max(torch.abs(self.dfdU(self.u[-1])))
        dt = self.C*self.dx/a
        self.Nt = np.ceil(self.T/dt).type(torch.int32) # TODO: sjekk at dette funker
        del a, dt
        _, self.dt = np.linspace(0, self.T, self.Nt, retstep=True)
    
    def godunov_flux(self, u_l, u_r):
        """
            Calculates the approximate Godunov flux using neural network.
        """
        if len(u_l.size())==0 and len(u_r.size())==0:
            u_l, u_r = u_l.expand(1), u_r.expand(1)
        U = torch.stack((u_l, u_r),dim=1)
        pred = []
        for u in U:
            A = self.net.forward(u.type(torch.float32))
            pred.append(A)
        pred = torch.tensor(pred)
        return pred

    def godunov(self):
        """
            Calculates the cell averages of next temporal step using Godunovs method.
        """
        C = self.dt/self.dx
        u = self.u[-1]
        u_next = torch.zeros(u.shape)
        u_next[1:-1] = u[1:-1] - C*(self.godunov_flux(u[1:-1], u[2:]) - self.godunov_flux(u[:-2], u[1:-1]))
        if (self.bnd_cond=='periodic'):
            u_next[0]  = u[0]  - C*(self.godunov_flux(u[0] , u[1]) - self.godunov_flux(u[-2] , u[0]))
            u_next[-1] = u[-1] - C*(self.godunov_flux(u[-1], u[1]) - self.godunov_flux(u[-2] , u[-1]))
        elif (self.bnd_cond=='dirichlet'):
            u_next[0]  = u[0]
            u_next[-1] = u[-1]
        elif (self.bnd_cond=='neumann'):
            u_next[0]  = u_next[1]  + (u[0]  - u[1])
            u_next[-1] = u_next[-2] + (u[-1] - u[-2])
        elif (self.bnd_cond=='robin'):
            a, b = 1, 1
            c = (a*u[1])/self.dx + (b - a/self.dx)*u[0]
            u_next[0] = (c - (a*u_next[1])/self.dx)/(b - a/self.dx)
            d, e = 1, 1
            f = (d*u[-1])/self.dx + (e - d/self.dx)*u[-2]
            g = 1 - (e*self.dx)/2
            u_next[-1] = u[-1] - g*u[-2] + g*u_next[-2]
        else:
            u_next[0]  = u[0]
            u_next[-1] = u[-1]
        self.u.append(u_next)

    @property
    def solve(self):
        t = 0
        pbar = tqdm(
            total=self.T, 
            desc='Solving progress', 
            bar_format = '{desc}: {percentage:3.0f}%{bar}Epoch: {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        while t<self.T:
            self.set_dt()
            t += self.dt
            if t>=self.T:
                self.dt -= (t-self.T)
                t = self.T
            self.godunov()
            pbar.update(self.dt)
        pbar.close()