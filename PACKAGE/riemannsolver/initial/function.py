"""
initial.function
================

Description
-----------
Provides initial function to use for initial data.

Usage
-----
Constructor (Dataset)
    __init__(self, func_name)


"""
import numpy as np

class InitialFunc:
    """
    Contains a selection of functions to use when testing the Godunovs method together
    with MLP implementation.
    """
    def __init__(self, func_name):
        self.func_name = func_name
        self.func_menu = ['linear',
                        'sine','sine_sq',
                        'exp','exp_sq',
                        'step', 'step_rev',
                        'heavi','heavi_rev','heavi_scaled','sine1D',
                        'watersplash', 'heavi_diag','circ_square']
        self.func = None
        self.set_func()
    
    def set_func(self):
        """
        If the function name is contained within the list of valid functions it is called,
        otherwise the zero-function is called.
        """
        if self.func_name in self.func_menu: eval("self."+self.func_name)
        else: self.zero

    @property
    def watersplash(self):
        def splash(mesh):
            r = np.sqrt(mesh[:,:,0]*mesh[:,:,0] + mesh[:,:,1]*mesh[:,:,1])
            return (np.exp(r)/(1 + np.exp(r)))*np.cos(4*np.pi*r)
        self.func = lambda mesh: splash(mesh)

    @property 
    def circ_square(self):
        def cs(mesh):
            x = np.logical_and(np.greater(mesh[:,:,1], 0.1),   np.less(mesh[:,:,1], 0.6))
            y = np.logical_and(np.greater(mesh[:,:,0], -0.25), np.less(mesh[:,:,0], 0.25))
            sq = np.round(1.0*np.logical_and(x, y))
            r = np.sqrt((mesh[:,:,1] + 0.45)**2 + mesh[:,:,0]**2)
            circ = (1-r/0.35)*np.less(r, 0.35)
            return (sq + circ)
        self.func = lambda mesh: cs(mesh)


    @property
    def heavi_diag(self):
        def diag(mesh):
            return np.greater(mesh[:,:,0]+mesh[:,:,1],0.0)*1.0
        self.func = lambda mesh: diag(mesh)

    @property
    def heavi(self):
        def heaviside(mesh):
            low_mesh = np.logical_and(np.greater_equal(mesh,-1.0), np.less(mesh,0.0))
            hig_mesh = np.logical_and(np.greater_equal(mesh,0.0), np.less_equal(mesh,1.0))

            low_x, low_y = low_mesh[:,:,0], low_mesh[:,:,1]
            hig_x, hig_y = hig_mesh[:,:,0], hig_mesh[:,:,1]

            func = (0.0)*np.logical_and(low_x, low_y) \
                + (0.0)*np.logical_and(low_x, hig_y) \
                + 1.0*np.logical_and(hig_x, low_y) \
                + 1.0*np.logical_and(hig_x, hig_y)
            return func
        self.func = lambda mesh: heaviside(mesh)
    
    @property
    def heavi_rev(self):
        def heaviside_rev(mesh):
            low_mesh = np.logical_and(np.greater_equal(mesh,-1.0), np.less(mesh,0.0))
            hig_mesh = np.logical_and(np.greater_equal(mesh,0.0), np.less_equal(mesh,1.0))

            low_x, low_y = low_mesh[:,:,0], low_mesh[:,:,1]
            hig_x, hig_y = hig_mesh[:,:,0], hig_mesh[:,:,1]

            func = (1.0)*np.logical_and(low_x, low_y) \
                + (1.0)*np.logical_and(low_x, hig_y) \
                + 0.0*np.logical_and(hig_x, low_y) \
                + 0.0*np.logical_and(hig_x, hig_y)
            return func
        self.func = lambda mesh: heaviside_rev(mesh)

    @property
    def heavi_scaled(self):
        def heaviside_scaled(mesh):
            low_mesh = np.logical_and(np.greater_equal(mesh,-1.0), np.less(mesh,0.0))
            hig_mesh = np.logical_and(np.greater_equal(mesh,0.0), np.less_equal(mesh,1.0))

            low_x, low_y = low_mesh[:,:,0], low_mesh[:,:,1]
            hig_x, hig_y = hig_mesh[:,:,0], hig_mesh[:,:,1]

            func = (-1.0)*np.logical_and(low_x, low_y) \
                + (-1.0)*np.logical_and(low_x, hig_y) \
                + 1.0*np.logical_and(hig_x, low_y) \
                + 1.0*np.logical_and(hig_x, hig_y)
            return func
        self.func = lambda mesh: heaviside_scaled(mesh)

    @property
    def sine1D(self):
        self.func = lambda mesh: np.sin(4*np.pi*mesh[:,:,0])

    @property
    def linear(self):
        self.func = lambda mesh: np.sum(mesh,axis=2)

    @property
    def sine(self):
        self.func = lambda mesh: np.sin(2*np.pi*np.sum(mesh,axis=2))

    @property
    def sine_sq(self):
        self.func = lambda mesh: np.sin(np.sum(mesh+mesh,axis=2))
    
    @property
    def exp(self):
        self.func = lambda mesh: np.exp(np.sum(mesh,axis=2))

    @property
    def exp_sq(self):
        self.func = lambda mesh: np.exp(np.sum(mesh*mesh,axis=2))

    @property
    def step(self):
        def step_func(mesh):
            low_mesh = np.logical_and(np.greater_equal(mesh,-1.0), np.less(mesh,0.0))
            hig_mesh = np.logical_and(np.greater_equal(mesh,0.0), np.less_equal(mesh,1.0))

            low_x, low_y = low_mesh[:,:,0], low_mesh[:,:,1]
            hig_x, hig_y = hig_mesh[:,:,0], hig_mesh[:,:,1]

            func = (-1.0)*np.logical_and(low_x, low_y) \
                + (-0.5)*np.logical_and(low_x, hig_y) \
                + 0.5*np.logical_and(hig_x, low_y) \
                + 1.0*np.logical_and(hig_x, hig_y)
            return func
        self.func = lambda mesh: step_func(mesh)

    @property
    def zero(self):
        self.func = lambda mesh: np.zeros(mesh.shape[0:2])
    
    @property
    def get_func(self):
        # need to set func wrt some functions.
        return self.func
    
    @property 
    def get_name(self):
        return self.func_name