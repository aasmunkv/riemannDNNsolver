"""
riemannsolver
=============

Provides
  1. Efficient solvers for 1-dimensional Riemann problems.
  2. Efficient solvers for 2-dimensional Riemann problems.
  3. Plotting tools for quick illustrations of training and solutions.

How to use the documentation
----------------------------
Todo: write this...


Available subpackages
---------------------
godunov
    Godunov-based Riemann solver for both 1- and 2-dimensional conservation laws.
dnn1d
    DNN based Riemann solver for 1-dimensional scalar conservation laws.
dnn2d
    DNN based Riemann solver for 2-dimensional scalar conservation laws.
flatplotlib
    Plotting tool for the Riemann solvers.
initial
    Initial functions for testing the solvers.
datagenerator
    Genrators for datasets to use when training the different DNNs.

"""
from .dnn1d import godunov as god_mlp1d, network as net_mlp1d
from .dnn2d import godunov as god_mlp2d, network as net_mlp2d, godunov_newParam as god_mlp2d_newParam
from .flatplotlib import netplot
from .datagenerator import data1d_newLoss, data1d, data2d, data2d_newParam
from .initial import function
from .godunov import godunov, godunov_2d
