"""
riemannsolver.datagenerator
===========================

Description
-----------
Provides data for training of dnn1d and dnn2d, within data1d and data2d, respectively.

Usage
-----
# EXAMPLE OF 1D DATA GENERATION
from riemannsolver.datagenerator import data1d
# define N (integer) and f (callable)
dat = data1d.Dataset(N=N, f=f)
dat.create
dat_train = dat.get_data

"""
from . import data1d
from . import data1d_newLoss
from . import data2d
