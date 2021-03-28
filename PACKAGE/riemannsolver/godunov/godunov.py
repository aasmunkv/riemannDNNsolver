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
    """
    This class is for producing the exact solution of the Godunov method in 1 dimension.
    """
    def __init__(self, f, dfdu, u0, 
            bnd_cond, xmin, xmax, Nx, 
            T=1.0, C=0.5):
        """
        Inputs:
            f - flux function
                type: callable
            dfdu - derivative of flux function
                type: callable
            U0 - initial function
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
            raise TypeError('f, dfdU and U0 is %s, %s and %s. All need to be functions.' % (type(f),type(dfdu),type(u0)))

        if not isinstance(bnd_cond, str):
            raise TypeError('bnd_cond is %s, not string. It needs to be string with boudnary condition.' % (type(bnd_cond)))
        elif bnd_cond not in ['dirichlet','neumann','periodic']:
            raise AssertionError("Argument 'bnd_cond' must be 'dirichlet','neumann' or 'periodic'.")

        self.f = lambda U: f(U)
        self.dfdu = lambda U: dfdu(U)
        self.U0 = lambda x: u0(x)
        
        self.bnd_cond = bnd_cond
        self.xmin, self.xmax, self.Nx = xmin, xmax, Nx

        self.god_flux_mesh_size = 10
        self.T = T
        self.C = C

        # self.x = torch.linspace(xmin, xmax, Nx)
        # self.dx = (xmax - xmin)/Nx
        
        self.x, self.dx = None, None
        self.set_dx()

        self.u0 = self.U0(self.x)
        self.u = [self.u0]

        self.dt, self.Nt = None,None
        self.dt_coeff = self.C*self.dx
        self.set_dt()
    
    def set_dx(self):
        """
        To set the x and dx variables.
        """
        self.x, self.dx = np.linspace(self.xmin, self.xmax, self.Nx, retstep=True)
        self.x = torch.tensor(self.x).type(torch.float64)

    def set_dt(self):
        """
        To set the Nt and dt variables.
        """
        a = torch.max(torch.abs(self.dfdu(self.u[-1])))
        dt = self.dt_coeff/a
        self.Nt = int(self.T//dt)
        _, self.dt = np.linspace(0, self.T, self.Nt, retstep=True)
    
    @property
    def solve(self):
        """
        To solve the conservation law of 1 spacial dimension using Godunovs method.

        NOTE: Observe that it is used a tolerance within the if-test. This is temporary
            so that the loop is ensured to quit.
        """
        t, cnt = 0, 0
        pbar = tqdm(
            total=self.T, 
            desc='Solving progress', 
            bar_format = '{desc}: {percentage:3.0f}%{bar}Column: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        while t < self.T:
            cnt += 1
            self.set_dt()
            t += self.dt
            if t>=self.T:
                self.dt -= (t-self.T)
                t = self.T
            self.godunov()
            pbar.update(self.dt)
        pbar.close()

    def godunov(self):
        """
        Godunov method in 1 spacial dimension.
        """
        u = self.u[-1]
        C = self.dt/self.dx
        u_next = torch.zeros(u.shape).cuda()
        u_next[1:-1] = u[1:-1].clone().detach().cuda() \
            - C*( self.godunov_flux(u[1:-1], u[2:]) - self.godunov_flux(u[:-2], u[1:-1]) 
            ).reshape([self.x.shape[0]-2]).cuda()
        
        if (self.bnd_cond=='periodic'):
            u_next[0]  = u[0].clone().detach().cuda() \
                - C*(self.godunov_flux(u[0],u[1]) - self.godunov_flux(u[-2],u[0])).cuda()
            u_next[-1] = u[-1].clone().detach().cuda() \
                - C*(self.godunov_flux(u[-1],u[1]) - self.godunov_flux(u[-2],u[-1])).cuda()
        
        elif (self.bnd_cond=='dirichlet'):
            u_next[0]  = u[0].cuda()
            u_next[-1] = u[-1].cuda()
        
        elif (self.bnd_cond=='neumann'):
            u_next[0]  = u_next[1].cuda()  + (u[0]  - u[1]).cuda()
            u_next[-1] = u_next[-2].cuda() + (u[-1] - u[-2]).cuda()
        
        else:
            u_next[0]  = u_next[0].cuda()
            u_next[-1] = u_next[-1].cuda()
        
        self.u.append(u_next.type(torch.float64).cpu())
    
    def godunov_flux(self, U_L, U_R):
        """
        To calculate the Godunov flux.
        """
        torch.cuda.empty_cache()
        U_L, U_R = torch.flatten(U_L).type(torch.float64).cuda(), torch.flatten(U_R).type(torch.float64).cuda()
        # Set up identity arr from 0 to 1 (to be scaled wrt U_L and U_R)
        arr = torch.linspace(0,1,self.god_flux_mesh_size, dtype=torch.float64).cuda()#self.god_flux_mesh_size)
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

