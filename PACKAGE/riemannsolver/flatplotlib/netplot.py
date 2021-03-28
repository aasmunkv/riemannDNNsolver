"""
flatplotlib.netplot
===================

Description
-----------
Provides plotting of results for mlp1d and mlp2d, as well as datagenerator.

Usage
-----
Constructor (Surface)
    __init__(self, mesh, z, fps=10)
Constructor (Curve)
    __init__(self, history, weights, god_network=None, god_exact=None)

Example 2D
----------
dataset = data2d.Dataset(...)
dataset.create
data = data.get_data
u, F, G = data
mesh = data.god.mesh
u = u[:,:-1,:-1]    # since size in axis 1 and 2 is filled with zeros
flt = netplot.Surface(mesh=mesh, z=u)
HTML(flt.get_anim_surface.to_html5_video()) # notebook spesific
"""
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Surface:
    """
    Yields suface plots and colormaps of three dimensional structures.
    This class is for use of results produces in the case of mlp2d.
    """
    def __init__(self, mesh, z, fps=10):
        """
        fps - frames per second
        fn - number of frames
        mesh - mesh containing x=mesh[:,:,0] adn y=mesh[:,:,1]
        z - height of wave, z.shape=mesh[:,:,0].shape=mesh[:,:,1].shape
        """
        self.x, self.y, self.z = mesh[:,:,0].numpy(), mesh[:,:,1].numpy(), z.numpy()

        self.fps, self.fn = fps, z.shape[0]

        self.anim_surface, self.anim_color = None, None
        self.color = None

        self.color_menu = self.get_color_menu
        
        self.set_color(0)
        self.maxVal = z.max()

        self.surface_plot
        self.color_plot


    def set_color(self, col_ind):
        if (0<=col_ind) and (col_ind<len(self.color_menu)) and isinstance(col_ind, int):
            self.color = self.color_menu[col_ind]
        else:
            self.color = self.color_menu[0]

    def set_maxval(self, val):
        self.maxVal = val


    @property
    def surface_plot(self):
        """
        Returns a surface animation plot of type
            matplotlib.animation.FuncAnimation
        """
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')



        ax.set_xlabel('x')
        ax.set_ylabel('y')

        axMax = np.max(self.z)+0.01*np.max(self.z)
        axMin = np.min(self.z)+0.01*np.min(self.z)
        ax.set_zlim(axMin,axMax)

        im = [ax.plot_surface(self.x,self.y,self.z[0] , cmap=self.color)]

        im_cb = im[0]

        cb = fig.colorbar(im_cb,ax=ax)

        def animate(i, z, im):
            im[0].remove()
            im[0] = ax.plot_surface(self.x,self.y, z[i], cmap=self.color, vmax=axMax)
            #ax.set_zlim(0,axMax)
            return (im[0],)

        self.anim_surface = animation.FuncAnimation(fig, animate, frames=self.fn, fargs=(self.z, im), interval=1000/self.fps, blit=True)
        plt.close()

    @property
    def color_plot(self):
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111)

        #ax.autoscale_view()

        im = ax.imshow(self.z[0].T, aspect='auto', cmap=self.color, extent=[-1,1,1,-1])#, norm=mplt.colors.LogNorm(vmax=self.maxVal))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cb = fig.colorbar(im, ax=ax)

        def init():
            im.set_data(self.z[0].T)
            return (im,)

        # animation function. This is called sequentially
        def animate(i):
            im.set_data(self.z[i].T)
            return (im,)

        # call the animator. blit=True means only re-draw the parts that have changed.
        self.anim_color = animation.FuncAnimation(fig, animate, frames=self.fn, init_func=init, interval=1000/self.fps)
        plt.close()

    def save_anim_surface(self, filename, folder="results/"):
        writer = self.anim_surface.save(
            folder+filename+'.gif', 
            writer=animation.writers['ffmpeg'](
                fps=self.fps, 
                metadata=dict(artist='aasmunkv'), 
                bitrate=8
            )
        )

    def save_anim_color(self, filename, folder="results/"):
        writer = self.anim_color.save(
            folder+filename+'.gif', 
            writer=animation.writers['ffmpeg'](
                fps=self.fps, 
                metadata=dict(artist='aasmunkv'), 
                bitrate=8
            )
        )

    @property
    def get_anim_surface(self):
        return self.anim_surface

    @property
    def get_anim_color(self):
        return self.anim_color

    @property
    def get_color_menu(self):
        color_menu = ['coolwarm',
                    'seismic',
                    'hsv',
                    'Spectral',
                    'Spectral_r']
        return color_menu

class Curve:
    """
    Curve
    =====

    For plotting training loss and validation loss mainly, but may be used for any result
    plotting of correct dimension.

    Example on usage
    ----------------
    # Retrieve 'hist' from network
    netplot.Curve(hist)
    """
    def __init__(self, history, weights, god_network=None, god_exact=None, god_godunov=None):
        """
        Inputs:
            hist - A list of torch tensors or numpy arrays to be plotted in same figure.        
        """
        self.hist = history
        self.weights = weights
        self.god_net = god_network
        self.god_ext = god_exact
        self.god_god = god_godunov

    def plot_history(self, destination, name, show=False):
        y_train, y_valid = self.hist
        y_len = len(y_train)
        x_train = torch.linspace(0, y_len, steps=y_len)
        x_valid = torch.linspace(start=0,end=y_len,steps=len(y_valid))

        plt.figure(figsize=(3,3))

        plt.plot(x_train, y_train,lw=0.8)
        plt.plot(x_valid, y_valid, '.-',lw=0.8)

        tick = np.arange(0, y_len+1, np.ceil(y_len/2).astype(np.int64))
        plt.xticks(ticks=tick,labels=tick)
        # plt.yticks(ticks=[10^(-2),2*10^(-2),3*10^(-2),4*10^(-2),6*10^(-2)])#,labels=[10^(-2),"","","",""])
        plt.legend(["Training","Validation"])
        #plt.ylim(bottom=6.2e-3,top=6.5e-2)
        plt.yscale("log")
        

        plt.tight_layout()

        destination = '.' if (destination == '') else destination
        plt.savefig(destination+'/'+'lossHist_'+name +'.pdf',format='pdf')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_weights(self, destination, name, show=False):
        hist_w = self.weights
        w_diff_restruct = [[] for _ in range(len(hist_w[0]))]
        for epoch in range(len(hist_w)-1):
            for layer in range(len(hist_w[epoch])):
                x = np.mean((hist_w[epoch+1][layer] - hist_w[epoch][layer])**2)#,axis=0)#/hist_w[epoch+1][layer].shape[0]
                for i in range(len(x.flatten())):
                    w_diff_restruct[layer].append(x.flatten()[i])
        w_diff_restruct = [np.array(i) for i in w_diff_restruct]

        plt.figure(figsize=(3,3))

        legs = []
        for i,w in enumerate(w_diff_restruct):
            x = np.linspace(1,len(hist_w),len(w))
            plt.plot(x,w,lw=0.8)
            legs.append("Layer "+str(i)+'-'+str(i+1))
        tick = np.arange(0, len(hist_w)+1, np.ceil(len(hist_w)/5).astype(np.int32))
        plt.xticks(ticks=tick,labels=tick)
        plt.yscale('log')
        plt.legend(legs)

        plt.tight_layout()

        destination = '.' if (destination == '') else destination
        plt.savefig(destination+'/'+'weightHist_'+name +'.pdf',format='pdf')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_exact(self, destination, name, show=False):
        if self.god_ext is None:
            raise UnboundLocalError("'self.god_ext' is None.")
        u = self.god_ext.u[-1]
        x = self.god_ext.x
        plt.plot(x, u, 'C0')
        plt.legend(['Exact'])
        
        destination = '.' if (destination == '') else destination
        plt.savefig(destination+'/'+'exact_'+name +'.pdf',format='pdf')
        if show==True:
            plt.show()
        else:
            plt.close()

    def plot_solution(self, destination, name, show=False):
        if self.god_net is None:
            raise UnboundLocalError("'self.god_net' is None.")
        u = self.god_net.u[-1]
        x = self.god_net.x
        plt.plot(x, u, 'C1.-')
        plt.legend(['Network'])
        
        destination = '.' if (destination == '') else destination
        plt.savefig(destination+'/'+'network_'+name +'.pdf',format='pdf')
        if show==True:
            plt.show()
        else:
            plt.close()

    def plot_godunov(self, destination, name, show=False):
        if self.god_god is None:
            raise UnboundLocalError("'self.god_god' is None.")
        u = self.god_god.u[-1]
        x = self.god_god.x
        plt.plot(x, u, 'C2--')
        plt.legend(['Godunov'])
        
        destination = '.' if (destination == '') else destination
        plt.savefig(destination+'/'+'godunov_'+name +'.pdf',format='pdf')
        if show==True:
            plt.show()
        else:
            plt.close()

    def plot_solution_exact(self, destination, name, show=False):
        if self.god_net is None or self.god_ext is None or self.god_god is None:
            raise UnboundLocalError("'god_net','god_ext' or 'god_god' equals None.")
        u = self.god_net.u[-1]
        x = self.god_net.x
        u_e = self.god_ext.u[-1]
        x_e = self.god_ext.x
        u_g = self.god_god.u[-1]
        x_g = self.god_god.x

        plt.figure(figsize=(3,3))

        plt.plot(x_e, u_e, 'C0',lw=0.8)
        plt.plot(x, u, 'C1.-',lw=0.8)
        plt.plot(x_g, u_g, 'C2--',lw=0.8)

        # tick_x = np.linspace(-1.0, 1.0, 5)
        # plt.xticks(ticks=tick_x,labels=tick_x,fontsize=15)
        # tick_y = np.round(np.linspace(min(u), max(u), 5),decimals=2)
        # plt.yticks(ticks=tick_y,labels=tick_y,fontsize=15)

        plt.legend(['Exact','Network','Godunov'])#,fontsize=15)
        #plt.xlabel('x')#,fontsize=15)
        #plt.ylabel('u')#,fontsize=15)

        plt.tight_layout()
        
        destination = '.' if (destination == '') else destination
        plt.savefig(destination+'/'+'exact_network_'+name+'.pdf',format='pdf')#,pad_inches=20)
        if show==True:
            plt.show()
        else:
            plt.close()

class Flux2D:
    def __init__(self, data, flux):
        """
            data: input data of dimension M x 6
                NOTE: do not send in target
        """
        self.data = data
        self.flux = flux

        self.anim = []
        self.anim_flux()

    def plot_cells(self, ind):
        try:
            data = self.data[ind]
        except IndexError: 
            raise IndexError('Index out of bounds...')
        if data.size(0)!=6:
            raise IndexError('Size of data must be 6, got %d...'%data.size(0))
        plt.imshow(data.reshape(3,2),cmap='gray')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def anim_flux(self):
        def anim(fx):
            max_fx = float(torch.max(fx))
            fig = plt.figure(figsize=(10,4))
            ax = plt.axes(xlim=(-1/3, 1/3), ylim=(-max_fx, max_fx*1.1))
            line, = ax.plot([], [], lw=1)
            ax.set_xlabel('y c [y_(j-1/2), y_(j+1/2)]')
            ax.set_ylabel('flux')
            # initialization function: plot the background of each frame
            def init():
                line.set_data(np.linspace(-1/3, 1/3, len(fx[0])), fx[0])
                return line,
            # animation function.  This is called sequentially
            def animate(i):
                line.set_data(np.linspace(-1/3, 1/3, len(fx[i])), fx[i])
                return line,
            # call the animator.  blit=True means only re-draw the parts that have changed.
            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=fx.size(0), interval=50, blit=True)
            return anim
        for fx in self.flux:
            self.anim.append(anim(fx=fx))
            plt.close()

    def plot_n_save_cells(self, data, name):
        fig = plt.figure(figsize=(6,5))
        plt.imshow(
            data.reshape(3,2),
            cmap='binary',
            vmin=float(torch.min(data)), 
            vmax=float(torch.max(data)),
            origin='lower'
        )
        plt.colorbar(ticks=np.linspace(float(torch.min(data)),float(torch.max(data)),4))
        plt.xticks([0,1],["$x_{i}$","$x_{i+1}$"])
        plt.yticks([0,1,2],["$y_{j-1}$","$y_j$","$y_{j+1}$"])
        plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=20)
        plt.tight_layout()
        plt.savefig('res/2dim_flux/cells_'+name+'.pdf', format='pdf')
        plt.show()

    def plot_n_save_flux(self, flux, name):
        """
        Takes in one argument, namely the flux tensor to plot.
        Then it taks the first, middle and last part and plots together in one plot.
        """
        max_fx = float(torch.max(flux))
        fig = plt.figure(figsize=(10,5))
        ax = plt.axes(xlim=(-1/3, 1/3), ylim=(-max_fx, max_fx*1.1))
        
        plt.plot(np.linspace(-1/3, 1/3, len(flux[0])), flux[0],'--',dashes=(2, 2),color='cyan')
        plt.plot(np.linspace(-1/3, 1/3, len(flux[flux.size(0)//2])), flux[flux.size(0)//2],'--',dashes=(7, 2),color='deepskyblue')
        plt.plot(np.linspace(-1/3, 1/3, len(flux[-1])), flux[-1],'-',color='blue')

        ytick = np.linspace(-0.9*max_fx,0.9*max_fx,3)
        plt.xticks([-1/3,0,1/3],["$y_{j-1/2}$","$y_j$","$y_{j+1/2}$"])
        plt.yticks(ytick,["{:.1e}".format(tck) for tck in ytick])

        plt.legend(["t=0",r"$t=\dfrac{T}{2}$","t=T"],loc='lower center')#, ncol=len(flux[0]))

        plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=20)
        plt.rc('legend', fontsize=20)
        plt.tight_layout()
        plt.savefig('res/2dim_flux/flux_'+name+'.pdf', format='pdf')
        plt.show()
