# Package
This directory contains the final package structure of both 1D and 2D solvers, as well as 
the testing of new loss function for 1 dimensional case. Otherwise, this folder contains 
different notebooks for different purposes, all of which has the main goal of producing 
sufficient amount of results for the thesis.

> Directory **riemannsolver** contains the package of which to use for producing results. 
	The package consist of multiple subpackages for each part of the implementation.
> * **dnn1d** contains the DNN based implementation for one-dimensional Riemann 
	problems.
> * **dnn2d** contains the DNN based implementation for two-dimensional Riemann 
	problems.
> * **initial** contains a selection of initial functions to use in package.
> * **flatplotlib** contains tools for plotting the results.
> * **godunov** contains Godunov solvers for 1-dimensional problems. It also 
	contains a fine mesh approximation of Godunovs method applied on 2 dimensional 
	problems.
> * **datagenerator** contains the code for generating datasets of both 1D and 2D 
        problems.
> Directory **res** contains subfolders for each produced result together with an 
	explainatory README.
> There are in total 6 Jupyter Notebook, namely
> * **create_reference_2dim**: for creating reference solutions of 2D experiments.
> * **create_results_1dim**: for creating results of 1D experiments.
> * **create_results_1dim_L1Loss**: for creating results of 1D experiments with the 
	extended L1-loss function.
> * **create_results_2dim**: for creating results of 2D experiments.
> * **create_results_2dim_flux**: for visualizing the non-constant flux of scalar 2D
	conservation laws.
> * **create_results_2dim_godunov**: for creating results of 2D experiments with Godunov's
	2D scheme.
