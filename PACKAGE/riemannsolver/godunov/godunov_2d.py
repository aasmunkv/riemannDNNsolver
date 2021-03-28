"""
exact.godunov_2d
=============

Description
-----------
Approximates the 2D scalar conservation law, by applying a fine-mesh 1D Godunov scheme in
both spatial directions.

Usage
-----
Constructor (Dataset)
    __init__(self, f, dfdU, g, dgdU, U0, 
        x_min, x_max, Nx, y_min, y_max, Ny, bnd_cond, T=None, C=0.5
    )

Contents
--------
The file contains two classes; one having the algorithm for fine mesh godunov scheme,
while the other is the one using the first mentioned class for solving the problem.

Other comment
-------------
After supervisor meeting with Ulrik, I was made aware of wrongful calculations of
reference solution. I was under impression that I needed to used fine-mesh algorithm,
however, this should only be used for calculation of training data for the DNN models.
The reference solutions should be calculated used e.g. 1000x1000 global mesh points.
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
    def __init__(self, f, dfdU, g, dgdU, U0, 
                x_min, x_max, Nx, 
                y_min, y_max, Ny,
                bnd_cond, T=None, C=0.5):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #self.finemesh_N = 20

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
    
    def godunov_flux(self, U_L, U_R):
        """
        To calculate the Godunov flux.
        """
        torch.cuda.empty_cache()
        U_L, U_R = torch.flatten(U_L).type(torch.float64).cuda(), torch.flatten(U_R).type(torch.float64).cuda()
        # Set up identity arr from 0 to 1 (to be scaled wrt U_L and U_R)
        arr = torch.linspace(0,1,1000, dtype=torch.float64).cuda()#self.god_flux_mesh_size)
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
        flux = ( (flux_min*cond[0] + flux_max*cond[1]).reshape(cond_shape) )
        del flux_min,flux_max
        return flux

    def god_flux(self):
        def flux(u):
            F = torch.zeros((u.size(0), u.size(1)-1))
            for i in range(0,F.size(0),10):
                F[i:i+10,:] = self.godunov_flux(u[i:i+10,:-1],u[i:i+10,1:]).reshape(F[i:i+10,:].size())
            return F
            '''
            F = torch.zeros((u.size(0)-2, u.size(1)-1))
            for i in range(F.size(0)):
                for j in range(F.size(1)):
                    finemesh_u0 = u[i:i+3,j:j+2].reshape(6)
                    solver = FineMeshSolver(
                        u0 = finemesh_u0,
                        N = self.finemesh_N,
                        f = self.f,
                        dfdu = self.dfdU,
                        g = self.g,
                        dgdu = self.dgdU,
                        T = self.dt, 
                        cuda_num=0
                    )
                    solver.solve
                    flux = torch.stack(solver.god_flux[0])
                    flux_of_interest = flux[:,self.finemesh_N:2*self.finemesh_N,flux.size(2)//2]
                    F[i,j] = torch.mean(flux_of_interest)
            return F
            '''
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
        u_G = u_F.rot90()
        #F, G = flux(u_F), flux(u_G).rot90(3)
        F, G = flux(u_F[1:-1]), flux(u_G[1:-1]).rot90(3)
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
            """
            Boundary values are equal to boundary of initial data.
            """
            u_next[:, 0] = self.u[0][:, 0].clone().detach().to(self.device)
            u_next[:,-1] = self.u[0][:,-1].clone().detach().to(self.device)
            u_next[0, :] = self.u[0][0, :].clone().detach().to(self.device)
            u_next[-1,:] = self.u[0][-1,:].clone().detach().to(self.device)
        elif self.bnd_cond=='neumann_zero':
            """
            Boundary's derivative is zero.
            """
            u_next[:, 0] = u_next[:, 1].clone().detach().to(self.device)
            u_next[:,-1] = u_next[:,-2].clone().detach().to(self.device)
            u_next[0, :] = u_next[1, :].clone().detach().to(self.device)
            u_next[-1,:] = u_next[-2,:].clone().detach().to(self.device)
        elif self.bnd_cond=='neumann':
            """
            Boundary's derivative is change at boundary of last solution.
            """
            u_next[:, 0] = u_next[:, 1].clone().detach().to(self.device) + C_x*(u[:, 0].clone().detach().to(self.device) - u[:, 1].clone().detach().to(self.device))
            u_next[:,-1] = u_next[:,-2].clone().detach().to(self.device) + C_x*(u[:,-1].clone().detach().to(self.device) - u[:,-2].clone().detach().to(self.device))
            u_next[0, :] = u_next[1, :].clone().detach().to(self.device) + C_y*(u[0, :].clone().detach().to(self.device) - u[1, :].clone().detach().to(self.device))
            u_next[-1,:] = u_next[-2,:].clone().detach().to(self.device) + C_y*(u[-1,:].clone().detach().to(self.device) - u[-2,:].clone().detach().to(self.device))
        else: # edge = neighbouring points, non-reflecting neumann
            """
            Boundary's derivative is zero.
            """
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

class FineMeshSolver:
    def __init__(self, u0, N, f, dfdu, g, dgdu, T, cuda_num = 0):
        """
        NOTE:
        For now I have decided to use hardcoded Courant coefficient instead of input.
        I have also decided to use hard coded Neumann boundary conditions temporarily.
        u0 is the six initial cell averages (sorted from upper left to lower right), 
        NOT the initial function.
        We do not use xmin, xmax for anything, so it is removed. To obtain values dx and 
        dy it is used hard coded domain of [-1,1]x[-1,1].
        Input N is telling us that each cell will be divided into N x N smaller cells.
        Hence we have 3N x 2N fine mesh.
        """
        torch.random.manual_seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        self.device = "cuda:0"#+str(cuda_num) # if torch.cuda.is_available() else "cpu"

        self.u0 = u0
        self.N = N
        self.f, self.dfdu = f, dfdu
        self.g, self.dgdu = g, dgdu
        self.T = T

        self.C = 0.5
        self.god_flux_mesh_size = 100
        self.x_size, self.y_size = 2*self.N, 3*self.N

        self.x, self.dx = np.linspace(-1, 1, self.x_size, retstep=True)
        self.y, self.dy = np.linspace(-1, 1, self.y_size, retstep=True)

        self.x = torch.tensor(self.x, dtype=torch.float64)
        self.y = torch.tensor(self.y, dtype=torch.float64)
        #self.mesh = torch.stack(torch.meshgrid(self.x, self.y),axis=2)

        self.dt = None

        self.god_flux = [[],[]]

        init_mesh = torch.zeros((self.y_size, self.x_size),dtype=torch.float64)
        init_mesh[:self.N,:self.N] = self.u0[0]
        init_mesh[:self.N,self.N:] = self.u0[1]
        init_mesh[self.N:2*self.N,:self.N] = self.u0[2]
        init_mesh[self.N:2*self.N,self.N:] = self.u0[3]
        init_mesh[2*self.N:,:self.N] = self.u0[4]
        init_mesh[2*self.N:,self.N:] = self.u0[5]
        self.u = [init_mesh]

        del init_mesh

    def set_dt(self):
        num = self.C*np.min((self.dx,self.dy))
        denom = torch.max( torch.sqrt(
            self.dfdu(self.u[-1])**2 + self.dgdu(self.u[-1])**2
        ) )
        dt = num/denom
        Nt = int(np.ceil(self.T/dt))
        _, self.dt = np.linspace(0, self.T, Nt, retstep=True)

        del num, denom, dt, Nt

    @property
    def solve(self):
        t = 0
        # pbar = tqdm(
        #     total=float(self.T), 
        #     desc='Solving progress', 
        #     bar_format = '{desc}: {percentage:3.0f}%{bar}Column: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        # )
        while t < self.T:
            self.set_dt()
            t += self.dt
            if t >= self.T:
                self.dt -= (t-self.T)
                t = self.T
            self.set_godunov_flux()
            self.godunov()
        #     pbar.update(float(self.dt))
        # pbar.close()

        del t#, pbar

    def godunov(self):
        god_flux_f, god_flux_g = self.god_flux[0][-1], self.god_flux[1][-1]
        Cx, Cy = self.dt/self.dx, self.dt/self.dy

        u_next = torch.zeros(self.u[-1].size()).to(self.device)
        u_next[:,:] = (self.u[-1][:,:]).clone().detach().to(self.device) \
            - Cx*( god_flux_f[:,1:] - god_flux_f[:,:-1]
            ).reshape((self.u[-1].size(0),self.u[-1].size(1))).clone().detach().to(self.device) \
            - Cy*( god_flux_g[1:,:] - god_flux_g[:-1,:]
            ).reshape((self.u[-1].size(0),self.u[-1].size(1))).clone().detach().to(self.device)
        
        # Neumann boundary conditions
        #u_next[:, 0] = u_next[:, 1].to(self.device) + (self.u[-1][:, 0].to(self.device) - self.u[-1][:, 1].to(self.device))
        #u_next[:,-1] = u_next[:,-2].to(self.device) + (self.u[-1][:,-1].to(self.device) - self.u[-1][:,-2].to(self.device))
        #u_next[0, :] = u_next[1, :].to(self.device) + (self.u[-1][0, :].to(self.device) - self.u[-1][1, :].to(self.device))
        #u_next[-1,:] = u_next[-2,:].to(self.device) + (self.u[-1][-1,:].to(self.device) - self.u[-1][-2,:].to(self.device))
            
        self.u.append(u_next.type(torch.float64).cpu())

        del god_flux_f, god_flux_g, Cx, Cy, u_next


    def set_godunov_flux(self):
        """
        NB!: Made with oposit thinking of indexing, thus u[y,x] (instead of u[x,y]).
        """
        # set up padded version of u[-1]
        u = torch.empty(self.u[-1].size(0)+2, self.u[-1].size(1)+2)
        u[1:-1, 1:-1] = self.u[-1]
        u[1:-1, 0] = u[1:-1, 1] # setting left bound
        u[1:-1,-1] = u[1:-1,-2] # setting right bound
        u[0, :] = u[1, :] # setting upper bound + corner
        u[-1,:] = u[-2,:] # setting lower bound + corner

        F = torch.empty(self.u[-1].size(0), self.u[-1].size(1)+1) # initialize F
        F[:,:] = self.godunov_flux(u[1:-1, :-1], u[1:-1, 1:], func=self.f).reshape(
            (self.u[-1].size(0), self.u[-1].size(1)+1)
        )
        self.god_flux[0].append(F)

        G = torch.empty(self.u[-1].size(0)+1, self.u[-1].size(1)) # initialize G
        G[:,:] = self.godunov_flux(u[:-1, 1:-1],u[1:, 1:-1], func=self.g).reshape(
            (self.u[-1].size(0)+1, self.u[-1].size(1))
        )
        self.god_flux[1].append(G)

        del F, G, u
    
    def godunov_flux(self, U_L, U_R, func):
        U_L, U_R = torch.flatten(U_L).type(torch.float64).to(self.device), torch.flatten(U_R).type(torch.float64).to(self.device)
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
        del flip, arr_mesh
        U_min = torch.min(U_L, U_R).reshape(U_L.size(),1)
        # Scale arr properly from min(U_L,U_R) to max(U_L,U_R)
        arr_scaled = torch.add(arr_mesh_flip*torch.abs(diff), U_min)
        del diff,arr_mesh_flip
        # Calculate function values
        arr_f = func(arr_scaled)
        del arr_scaled, func
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
        flux = (flux_min*cond[0] + flux_max*cond[1]).reshape(cond_shape)
        del flux_min,flux_max, cond, cond_shape
        return flux
