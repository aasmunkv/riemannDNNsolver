"""
datagenerator.data2d
====================

Description
-----------
Provides data for training of mlp2d.

Usage
-----
Constructor (Dataset)
    __init__(self, x, y, f, dfdu, g, dgdu, T, C, init_name)

Note
----
Class 'Godunov' is not ment for external use, only internal in module.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path

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
    def __init__(self, f, dfdU, g, dgdU, U0, 
                x_min, x_max, Nx, 
                y_min, y_max, Ny,
                bnd_cond, network, T=None, C=0.5):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.f = lambda U: f(U)
        self.dfdU = lambda U: dfdU(U)
        self.g = lambda U: g(U)
        self.dgdU = lambda U: dgdU(U)
        self.U0 = lambda mesh: torch.from_numpy(U0(mesh.numpy()))#lambda mesh: torch.tensor(U0(mesh))

        self.x_min, self.x_max, self.Nx = x_min, x_max, Nx
        self.y_min, self.y_max, self.Ny = y_min, y_max, Ny

        self.bnd_cond = bnd_cond

        self.C = C
        self.T = T
        self.t = 0 # time counter adding the time which has been iterated through

        self.x, self.dx = np.linspace(self.x_min, self.x_max, self.Nx, retstep=True)
        self.y, self.dy = np.linspace(self.y_min, self.y_max, self.Ny, retstep=True)

        y = torch.linspace(self.y_min, self.y_max, self.Ny, dtype=torch.float64)
        x = torch.linspace(self.x_min, self.x_max, self.Nx, dtype=torch.float64)
        y_mesh,x_mesh = torch.meshgrid(y,x)
        self.mesh = torch.stack((x_mesh,y_mesh),axis=2)

        self.net = network

        self.u0 = self.U0(self.mesh)
        self.u = [self.u0]

        self.T = torch.max(torch.sqrt( self.dfdU(self.u0)**2 + self.dgdU(self.u0)**2 ))
        if T is not None:
            if T < self.T: self.T = T
        
        self.dt = None        

    def set_dt(self):
        a = torch.max(
            torch.sqrt(self.dfdU(self.u[-1])**2 + self.dgdU(self.u[-1])**2)
        )
        dt = self.C*np.min((self.dx,self.dy))/a
        Nt = int(np.ceil(self.T/dt))
        _, self.dt = np.linspace(0, self.T, Nt, retstep=True)

    def god_flux(self):
        """
        Calculation of flux using network.
        """
        def flux(u):
            F = torch.zeros((u.size(0)-2, u.size(1)-1))
            for i in range(F.size(0)):
                for j in range(F.size(1)):
                    F_inp = u[i:i+3,j:j+2].reshape(6)
                    F[i,j] = self.net.forward(F_inp.type(torch.float32))
            return F
        u_pad = torch.cat(
            (
                self.u[-1][:,-2].reshape(self.u[-1].size(0),1),
                self.u[-1],
                self.u[-1][:,1].reshape(self.u[-1].size(0),1)
            ), dim=1
        )
        u_F = torch.cat(
            (
                u_pad[-2,:].reshape(1, u_pad.size(1)),
                u_pad,
                u_pad[1,:].reshape(1, u_pad.size(1))
            ), dim=0
        )
        del u_pad
        u_G = u_F.rot90()
        F, G = flux(u_F), flux(u_G).rot90(3)
        del u_F, u_G
        return (F,G)
        


    def godunov(self):
        """
        Index of u is opposite of what one might think: u[y_ind,x_ind]
        """
        u = self.u[-1].type(torch.float64)
        C_x, C_y = self.dt/self.dx, self.dt/self.dy
        
        god_flux_f, god_flux_g = self.god_flux()

        u_next = torch.zeros(u.shape).to(self.device)
        u_next[1:-1,1:-1] = (u[1:-1,1:-1]).clone().detach().to(self.device) \
            - C_x*( god_flux_f[1:-1,2:-1] - god_flux_f[1:-1,1:-2]
            ).reshape([self.u[-1].size(0)-2,self.u[-1].size(1)-2]).clone().detach().to(self.device) \
            - C_y*( god_flux_g[2:-1,1:-1] - god_flux_g[1:-2,1:-1]
            ).reshape([self.u[-1].size(0)-2,self.u[-1].size(1)-2]).clone().detach().to(self.device)

        
        if self.bnd_cond=='periodic':
            u_next[1:-1, 0] = u[1:-1,0].clone().detach().to(self.device) \
                - C_x*( god_flux_f[1:-1,1] - god_flux_f[1:-1,0] ).clone().detach().to(self.device) \
                - C_y*( god_flux_g[2:-1,0] - god_flux_g[1:-2,0] ).clone().detach().to(self.device)
                
            u_next[1:-1,-1] = u[1:-1,-1].clone().detach().to(self.device) \
                - C_x*( god_flux_f[1:-1,-1] - god_flux_f[1:-1,-2] ).clone().detach().to(self.device) \
                - C_y*( god_flux_g[2:-1,-1] - god_flux_g[1:-2,-1] ).clone().detach().to(self.device)
                
            u_next[0, 1:-1] = u[0,1:-1].clone().detach().to(self.device) \
                - C_x*( god_flux_f[0,2:-1] - god_flux_f[0,1:-2] ).clone().detach().to(self.device) \
                - C_y*( god_flux_g[1,1:-1] - god_flux_g[0,1:-1] ).clone().detach().to(self.device)
                
            u_next[-1,1:-1] = u[-1,1:-1].clone().detach().to(self.device) \
                - C_x*( god_flux_f[-1,2:-1] - god_flux_f[-1,1:-2] ).clone().detach().to(self.device) \
                - C_y*( god_flux_g[-1,1:-1] - god_flux_g[-2,1:-1] ).clone().detach().to(self.device)
                
            u_next[0,0] = (u_next[0,1] + u_next[1,0])/2
            u_next[0,-1] = (u_next[0,-2] + u_next[1,-1])/2
            u_next[-1,0] = (u_next[-1,1] + u_next[-2,0])/2
            u_next[-1,-1] = (u_next[-1,-2] + u_next[-2,-1])/2
        elif self.bnd_cond=='dirichlet':
            u_next[:, 0] = u[:, 0].clone().detach().to(self.device)
            u_next[:,-1] = u[:,-1].clone().detach().to(self.device)
            u_next[0, :] = u[0, :].clone().detach().to(self.device)
            u_next[-1,:] = u[-1,:].clone().detach().to(self.device)
        elif self.bnd_cond=='neumann':
            u_next[:, 0] = u_next[:, 1].clone().detach().to(self.device) + (u[:, 0].clone().detach().to(self.device) - u[:, 1].clone().detach().to(self.device))
            u_next[:,-1] = u_next[:,-2].clone().detach().to(self.device) + (u[:,-1].clone().detach().to(self.device) - u[:,-2].clone().detach().to(self.device))
            u_next[0, :] = u_next[1, :].clone().detach().to(self.device) + (u[0, :].clone().detach().to(self.device) - u[1, :].clone().detach().to(self.device))
            u_next[-1,:] = u_next[-2,:].clone().detach().to(self.device) + (u[-1,:].clone().detach().to(self.device) - u[-2,:].clone().detach().to(self.device))
        else: # edge = neighbouring points, non-reflecting neumann
            u_next[:, 0] = u_next[:, 1].clone().detach().to(self.device)
            u_next[:,-1] = u_next[:,-2].clone().detach().to(self.device)
            u_next[0, :] = u_next[1, :].clone().detach().to(self.device)
            u_next[-1,:] = u_next[-2,:].clone().detach().to(self.device)
        
        self.u.append(u_next.type(torch.float64).cpu())


    @property
    def solve(self):
        t = 0
        pbar = tqdm(
            total=float(self.T), 
            desc='Solving progress', 
            bar_format = '{desc}: {percentage:3.0f}%{bar}Column: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        while t < self.T:
            self.set_dt()
            t += self.dt
            if t>=self.T:
                self.dt -= (t-self.T)
                t = self.T
            self.godunov()
            pbar.update(float(self.dt))
        pbar.close()
        self.t += t


    @property
    def get_u(self):
        return torch.stack(self.u)


"""
OLD CODE FOR CALCULATING THE GODUNOV FLUX USING NETWORK
=======================================================
    def god_flux(self, u):
        row_num, col_num = u.size(0), u.size(1)

        F = torch.zeros((row_num-2, col_num-1))
        G = torch.zeros((row_num-1, col_num-2))

        F_inp = torch.zeros((row_num-2, col_num-1, 6))
        G_inp = torch.zeros((row_num-1, col_num-2, 6))

        for i in range(row_num-2):
            for j in range(col_num-1):
                F_inp[i,j] = u[i:i+3,j:j+2].reshape(6)
                F[i,j] = self.net.forward(F_inp[i,j].type(torch.float32))

        for i in range(row_num-1):
            for j in range(col_num-2):
                G_inp[i,j] = u[i:i+2,j:j+3].rot90(3).reshape(6)
                G[i,j] = self.net.forward(G_inp[i,j].type(torch.float32))

        return (F,G)
"""