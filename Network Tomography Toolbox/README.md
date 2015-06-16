*****************************************************************************
* Matlab code and data to reproduce results from the paper                  *
* "Super-Resolution from Unregistered and Totally Aliased Signals           *
*  using Subspace Methods"                                                  *
* Patrick Vandewalle, Luciano Sbaiz, Joos Vandewalle and Martin Vetterli    *
* available at http://lcavwww.epfl.ch/reproducible_research/VandewalleSVV06/*
*                                                                           *
* Copyright (C) 2006 Laboratory of Audiovisual Communications (LCAV),       *
* Ecole Polytechnique Federale de Lausanne (EPFL),                          *
* CH-1015 Lausanne, Switzerland.                                            *
*                                                                           *
* This program is free software; you can redistribute it and/or modify it   *
* under the terms of the GNU General Public License as published by the     *
* Free Software Foundation; either version 2 of the License, or (at your    *
* option) any later version. This software is distributed in the hope that  *
* it will be useful, but without any warranty; without even the implied     *
* warranty of merchantability or fitness for a particular purpose.          *
* See the GNU General Public License for more details                       *
* (enclosed in the file GPL).                                               *
*                                                                           *
* Latest modifications: August 25, 2006.                                    *
*****************************************************************************
        

* Abbreviations used in this document:
* ER: easily reproducible, taking less than 15 minutes
* CR: complexly reproducible, taking more than 15 minutes
* NR: not reproducible (images of original data, or manually drawn figures)

*************************************
* FIGURES AND TABLES FROM THE PAPER *
*************************************

Figure 1:  ER - can be reproduced using figure_1.m 
Figure 2:  ER - can be reproduced using figure_2.m
Figure 3:  NR - manually drawn figure
Figure 4:  left: ER - can be reproduced using figure_4.m
           right: NR - manually drawn figure
Figure 5:  ER - can be reproduced using figure_5.m
Figure 6:  NR - manually drawn figure
Figure 7:  NR - manually drawn figure
Figure 8:  ER - can be reproduced using figure_8.m
Figure 9:  NR - manually drawn figure
Figure 10: ER - can be reproduced using figure_10.m
Figure 11: ER - can be reproduced using figure_11.m
Figure 12: NR - manually drawn diagram
Figure 13: ER - can be reproduced using figure_13.m
Figure 14: ER - can be reproduced using figure_14.m
Figure 15: CR - can be reproduced using figure_15.m
                note that because of the long duration of these simulations,
                it might be better to compute the results using each method
                separately, saving the data, and then plotting the results
Figure 16: CR - can be reproduced using figure_16.m
                note that because of the long duration of these simulations,
                it might be better to compute the results using each method
                separately, saving the data, and then plotting the results
Figure 17: CR - can be reproduced using figure_17.m
                note that because of the long duration of these simulations,
                it might be better to compute the results using each method
                separately, saving the data, and then plotting the results
Figure 18: CR - can be reproduced using figure_18.m
                note that because of the long duration of these simulations,
                it might be better to compute the results for each of the
                images separately

Table 1:   NR - complexity analysis of the different algorithms


*************************************
* DATA AND IMAGES USED IN THE PAPER *
*************************************

images/castle.tif    : high resolution image used in the simulations
images/castle2.tif   : high resolution image used as aliasing example
images/cathedral.jpg : high resolution image used in the simulations
images/guitar.tif    : high resolution image used as aliasing example


**********************************************
* ADDITIONAL FILES USED IN THE ABOVE SCRIPTS *
**********************************************

basis_function.m     : generate a basis function sampled at certain 
                       locations
basis_function_2D.m  : generate a 2D basis function sampled at certain 
                       locations
estimate_shift.m     : estimate shifts between images with pairwise
                       alignments using only low frequencies
evaluateError_projection.m : evaluate the objective function
                       for the projection method
evaluateError_projection_2D.m : evaluate the objective function for the 
		       projection method on 2D images
evaluateError_projection_fourier.m : evaluate the objective function for the 
		       projection method using the Fourier basis
evaluateError_rank.m : evaluate the objective function
                       for the rank method
evaluateError_rank_2D.m  : evaluate the objective function
                       for the rank method on 2D images
generate_signal.m    : generate multiple sets of samples 
                       from a random or real signal
generate_signal_2D.m : generate multiple undersampled images from a real image
ndgrid_new.m         : generate a matrix enumerating all possible combinations
                       of a number of indices
reconstruct.m        : reconstruct signal coefficients from multiple sets of
                       aliased samples and their relative offsets
reconstruct_2D.m     : reconstruct image coefficients from multiple
                       aliased images and their relative offsets
reconstruct_2D_fourier.m : reconstruct image Fourier coefficients from multiple
                       aliased images and their relative offsets 
                       (more efficient than the general solution)
results_N_projection_full.m : used for figure 15
                       run simulations to test the projection-based method
                       on three sets of 1D samples
results_N_projection_heuristics.m : used for figure 15
                       run simulations to test the heuristic projection-based 
                       method on three sets of 1D samples
results_N_rank_full.m : used for figure 15
                       run simulations to test the rank-based method
                       on three sets of 1D samples
results_projection_full.m : used for figure 14
                       run simulations to test the projection-based method
                       on three sets of 1D samples
results_projection_heuristics.m : used for figure 14
                       run simulations to test the heuristic projection-based 
                       method on three sets of 1D samples
results_rank_full.m : used for figure 14
                       run simulations to test the rank-based method
                       on three sets of 1D samples
solve_projection_2D.m : pairwise alignment method to compute the offsets
                       using the projection-based method on images
solve_projection_2D_2step.m : hierarchical method to compute the offsets
                       using the projection-based method on images
solve_projection_fourier_full.m : method to compute the offsets
                       using the projection-based method on Fourier signals
solve_projection_fourier_heuristics.m : method to compute the offsets
                       using the heuristic projection-based method 
		       on Fourier signals
solve_projection_full_eval_points.m : method to compute the offsets
                       using the projection-based method; evaluation on a 
		       modified regular grid to get 0 value at correct offset
solve_projection_full.m : method to compute the offsets
                       using the projection-based method
solve_projection_heuristics.m : method to compute the offsets
                       using the heuristic projection-based method
solve_rank_2D.m      : pairwise alignment method to compute the offsets
                       using the rank-based method on images
solve_rank_full_eval_points.m : method to compute the offsets
                       using the rank-based method; evaluation on a 
		       modified regular grid to get 0 value at correct offset
solve_rank_full.m    : method to compute the offsets using the rank-based 
                       method
sr_2D.m              : generate a set of low resolution images and compute the
		       registration parameters and a high resolution 
		       reconstructed image from them
sr.m                 : generate multiple low resolution sets of samples 
		       and compute the registration parameters and a high 
		       resolution reconstructed signal from them
