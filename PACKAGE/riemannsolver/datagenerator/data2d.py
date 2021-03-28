"""
datagenerator.data2d
========================
NOTE
----
In this file I will develop a 2D datagenerator based on datagenerator.data2d.
As of now I think it will contain 2 classes, one of which takes care of the fine mesh 
calculations, while the other takes care of the data generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path
import multiprocessing
import os

in_jupy = False
try:
    cfg = get_ipython().__class__.__name__
    if cfg == 'ZMQInteractiveShell': in_jupy = True
except NameError: in_jupy = False
if in_jupy: from tqdm import tqdm_notebook as tqdm # for notebook
else: from tqdm import tqdm # for terminal

class Dataset:
    def __init__(self, M, N, f, dfdu, g, dgdu):
        """
        NOTE:
        Size of dataset = M x 7
        Contents of dataset hard coded to be in [-1,1]
        """
        self.M = M 
        self.f, self.dfdu = f, dfdu
        self.g, self.dgdu = g, dgdu

        self.N = N
        self.T = None

        self.data = torch.zeros(self.M, 7)
        self.data [:,:-1] = torch.randn(self.M,6)
        self.flux = []

    def set_T(self, u0):
        """
        Courant coefficient = 0.5
        """
        dx = 1.0 # since we have 2 cells in domain [-1,1] (hard coded in FineMeshSolver)
        dy = 2/3 # since we have 3 cells in domain [-1,1] (hard coded in FineMeshSolver)
        num = np.min((dx,dy))
        denom = 2*torch.max(np.sqrt(self.dfdu(u0)**2 + self.dgdu(u0)**2))
        self.T = float(num/denom)

    def set_F(self, flux, ind):
        """
        Summing up the flux.
        """
        flux_of_interest = flux[:,self.N:2*self.N,flux.size(2)//2]
        self.flux.append(flux_of_interest)
        self.data[ind, -1] = torch.mean(flux_of_interest)#, dtype=torch.float64)


    def create(self, cuda_num = 0):
        for i, u0 in enumerate(self.data[:,:-1]):
            print("Data",i+1,"of",self.M)
            self.set_T(u0=u0)
            solver = FineMeshSolver(
                u0 = u0,
                N = self.N,
                f = self.f,
                dfdu = self.dfdu,
                g = self.g,
                dgdu = self.dgdu,
                T = self.T,
                cuda_num=cuda_num
            )
            solver.solve
            flux = torch.stack(solver.god_flux[0])
            self.set_F(flux=flux, ind=i)
    
    def save(self, destination, filename):
        if destination == '':
            destination = '.'

        if self.data is None:
            raise UnboundLocalError("'self.data' is None.")

        if not os.path.exists(destination):
            raise FileNotFoundError("The given directory/folder does not exist.")

        if os.path.exists(destination+'/'+'data_'+str(self.M)+'_'+filename+'.pt'):
            raise FileExistsError("The filename already exists.")

        torch.save(self.data, destination+'/'+'data_'+str(self.M)+'_'+filename+'.pt')
        print("Data is saved in "+destination+'/'+'data_'+str(self.M)+'_'+filename+'.pt')
    
    def load(self, destination, filename):
        if not os.path.exists(destination):
            raise FileNotFoundError("The directory doies not exist.")

        if not os.path.exists(destination+'/'+filename+'.pt'):
            raise FileExistsError("The file does not exist.")

        self.data = torch.load(destination+'/'+filename+'.pt')
        print("Data is loaded from "+destination+'/'+filename)



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
        init_mesh[:self.N:self.N] = self.u0[0]
        init_mesh[:self.N,self.N:] = self.u0[1]
        init_mesh[self.N:2*self.N,:self.N] = self.u0[2]
        init_mesh[self.N:2*self.N,self.N:] = self.u0[3]
        init_mesh[2*self.N:,:self.N] = self.u0[4]
        init_mesh[2*self.N:,self.N:] = self.u0[5]
        self.u = [init_mesh]

    def set_dt(self):
        nom = self.C*np.min((self.dx,self.dy))
        denom = torch.max( torch.sqrt(
            self.dfdu(self.u[-1])**2 + self.dgdu(self.u[-1])**2
        ) )
        dt = nom/denom
        Nt = int(np.ceil(self.T/dt))
        _, self.dt = np.linspace(0, self.T, Nt, retstep=True)

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
            if t >= self.T:
                self.dt -= (t-self.T)
                t = self.T
            self.set_godunov_flux()
            self.godunov()
            pbar.update(float(self.dt))
        pbar.close()

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
        del F

        G = torch.empty(self.u[-1].size(0)+1, self.u[-1].size(1)) # initialize G
        G[:,:] = self.godunov_flux(u[:-1, 1:-1],u[1:, 1:-1], func=self.g).reshape(
            (self.u[-1].size(0)+1, self.u[-1].size(1))
        )
        self.god_flux[1].append(G)
        del G, u
    
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
        del flip
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
        del flux_min,flux_max
        return flux


"""
Old code for setting the godunov flux using batch iteration
===========================================================
    def set_godunov_flux(self):
        '''
        Calculates the godunov fluxes in x and y direction.
        Inputs: None
        Outputs:
            returns (flux_f,flux_g) and sets their respective global variable.
        '''
        # set last calculated u-values
        u = self.u[-1]

        step_y = self.batch_size if self.y_size>self.batch_size+1 else (self.y_size//2)+1
        step_x = self.batch_size if self.x_size>self.batch_size+1 else (self.x_size//2)+1
        batch_ind_y = torch.cat(
            (torch.arange(0, self.y_size-1, step=step_y).type(torch.float64),torch.tensor([self.y_size-1], dtype=torch.float64))
        )
        batch_ind_x = torch.cat(
            (torch.arange(0, self.x_size-1, step=step_x).type(torch.float64),torch.tensor([self.x_size-1], dtype=torch.float64))
        )
        del step_y, step_x

        # initialize godunov flux in x direction
        god_flux_f = torch.zeros((self.x_size+1),self.y_size) # Adding ghost cell in x-direction
        # initialize godunov flux in y direction
        god_flux_g = torch.empty(self.x_size,(self.y_size+1)) # Adding ghost cell in y-direction
        for i in range(len(batch_ind_y)-1):
            for j in range(len(batch_ind_x)-1):
                # calculate flux values in x direction
                y_ind1,y_ind2 = int(batch_ind_y[i]),int(batch_ind_y[i+1])
                x_ind1,x_ind2 = int(batch_ind_x[j]),int(batch_ind_x[j+1])

                god_flux_f[x_ind1+1:x_ind2+1,y_ind1:y_ind2] = self.godunov_flux(u[x_ind1:x_ind2,y_ind1:y_ind2], u[x_ind1+1:x_ind2+1,y_ind1:y_ind2], func=self.f).reshape((x_ind2-x_ind1,y_ind2-y_ind1)).to(self.device)
                god_flux_f[-1,y_ind1:y_ind2] = self.godunov_flux(u[-2,y_ind1:y_ind2], u[0,y_ind1:y_ind2], func=self.f).reshape((1,y_ind2-y_ind1)).to(self.device) # periodicity specific
                god_flux_f[0, y_ind1:y_ind2] = self.godunov_flux(u[-1,y_ind1:y_ind2], u[1,y_ind1:y_ind2], func=self.f).reshape((1,y_ind2-y_ind1)).to(self.device) # periodicity specific
                if (batch_ind_y[i+1]==self.y_size-1):
                    god_flux_f[1:-1,-1] = self.godunov_flux(u[:-1,-1], u[1:,-1], func=self.f).reshape((self.x_size-1)).to(self.device)
                    god_flux_f[-1,  -1] = self.godunov_flux(u[-2, -1], u[0, -1], func=self.f).to(self.device) # periodicity specific
                    god_flux_f[0,   -1] = self.godunov_flux(u[-1, -1], u[1, -1], func=self.f).to(self.device) # periodicity specific
                del y_ind1,y_ind2,x_ind1,x_ind2
                # calculate flux values in y direction
                y_ind1,y_ind2 = int(batch_ind_y[i]),int(batch_ind_y[i+1])
                x_ind1,x_ind2 = int(batch_ind_x[j]),int(batch_ind_x[j+1])

                god_flux_g[x_ind1:x_ind2,y_ind1+1:y_ind2+1] = self.godunov_flux(u[x_ind1:x_ind2,y_ind1:y_ind2], u[x_ind1:x_ind2,y_ind1+1:y_ind2+1], func=self.g).reshape((x_ind2-x_ind1,y_ind2-y_ind1)).to(self.device)
                god_flux_g[x_ind1:x_ind2,-1] = self.godunov_flux(u[x_ind1:x_ind2,-2], u[x_ind1:x_ind2,0], func=self.g).reshape((x_ind2-x_ind1)).to(self.device) # periodicity specific
                god_flux_g[x_ind1:x_ind2, 0] = self.godunov_flux(u[x_ind1:x_ind2,-1], u[x_ind1:x_ind2,1], func=self.g).reshape((x_ind2-x_ind1)).to(self.device) # periodicity specific
                if (batch_ind_x[j+1]==self.x_size-1):
                    god_flux_g[-1,1:-1] = self.godunov_flux(u[-1,:-1], u[-1,1:], func=self.g).reshape((1,self.y_size-1)).to(self.device)
                    god_flux_g[-1,  -1] = self.godunov_flux(u[-1, -2], u[-1, 0], func=self.g).to(self.device) # periodicity specific
                    god_flux_g[-1,   0] = self.godunov_flux(u[-1, -1], u[-1, 1], func=self.g).to(self.device) # periodicity specific
                del y_ind1,y_ind2,x_ind1,x_ind2
        print(god_flux_f[int(batch_ind_x[1]-1):int(batch_ind_x[1]+2), int(batch_ind_y[1]-1):int(batch_ind_y[1]+2)])
        del u, batch_ind_y, batch_ind_x
        # save godunov fluxes
        self.god_flux[0].append(god_flux_f)
        self.god_flux[1].append(god_flux_g)
        del god_flux_f, god_flux_g

Code for flexible boundary conditions in godunov()
==================================================
if self.bnd_cond=='periodic':
    u_next[1:-1, 0] = torch.tensor(u[1:-1,0],dtype=torch.float64).to(self.device) \
        - Cx*( god_flux_f[2:-1,0] - god_flux_f[1:-2, 0] ).to(self.device) \
        - Cy*( god_flux_g[1:-1,1] - god_flux_g[1:-1,-1] ).to(self.device)
        
    u_next[1:-1,-1] = torch.tensor(u[1:-1,-1],dtype=torch.float64).to(self.device) \
        - Cx*( god_flux_f[2:-1,-1] - god_flux_f[1:-2,-1] ).to(self.device) \
        - Cy*( god_flux_g[1:-1, 0] - god_flux_g[1:-1,-1] ).to(self.device)
        
    u_next[0, 1:-1] = torch.tensor(u[0,1:-1],dtype=torch.float64).to(self.device) \
        - Cx*( god_flux_f[1,1:-1] - god_flux_f[-1,1:-1] ).to(self.device) \
        - Cy*( god_flux_g[0,2:-1] - god_flux_g[0,1:-2 ] ).to(self.device)
        
    u_next[-1,1:-1] = torch.tensor(u[-1,1:-1],dtype=torch.float64).to(self.device) \
        - Cx*( god_flux_f[0, 1:-1] - god_flux_f[-1,1:-1] ).to(self.device) \
        - Cy*( god_flux_g[-1,2:-1] - god_flux_g[-1,1:-2] ).to(self.device)
        
    u_next[0,0] = (u_next[0,1] + u_next[1,0])/2
    u_next[0,-1] = (u_next[0,-2] + u_next[1,-1])/2
    u_next[-1,0] = (u_next[-1,1] + u_next[-2,0])/2
    u_next[-1,-1] = (u_next[-1,-2] + u_next[-2,-1])/2
elif self.bnd_cond=='dirichlet':
    u_next[:, 0] = u[:, 0].to(self.device)
    u_next[:,-1] = u[:,-1].to(self.device)
    u_next[0, :] = u[0, :].to(self.device)
    u_next[-1,:] = u[-1,:].to(self.device)
elif self.bnd_cond=='neumann':
    u_next[:, 0] = u_next[:, 1].to(self.device) + (u[:, 0].to(self.device) - u[:, 1].to(self.device))
    u_next[:,-1] = u_next[:,-2].to(self.device) + (u[:,-1].to(self.device) - u[:,-2].to(self.device))
    u_next[0, :] = u_next[1, :].to(self.device) + (u[0, :].to(self.device) - u[1, :].to(self.device))
    u_next[-1,:] = u_next[-2,:].to(self.device) + (u[-1,:].to(self.device) - u[-2,:].to(self.device))
else: # edge = neighbouring points, non-reflecting neumann
    u_next[:, 0] = u_next[:, 1].to(self.device)
    u_next[:,-1] = u_next[:,-2].to(self.device)
    u_next[0, :] = u_next[1, :].to(self.device)
    u_next[-1,:] = u_next[-2,:].to(self.device)
"""
