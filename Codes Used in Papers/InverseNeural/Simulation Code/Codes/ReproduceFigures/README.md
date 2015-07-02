Introduction
===================
This repository contains the code to reproduce the figures for the paper [Inverse Neural](http://rr.epfl.ch/paper/KSV2015). In what follows, we describe how one can generate the figures for the paper and the ones in the supplementary materials.

Reproducing the Figures
===================
The codes are written in *Python*. The required libraries and dependencies are listed further down. Here, we discuss how to run the code. In principle, there are several to get the results shown in the paper, one of which is running the algorithm itself and the others involve generating data and evaluating the performance of the algorithm. The steps are listed below:

1. Running the inference algorithm (Cross Correlogram, GLM and NeuInf)
2. Transforming the resulting *analog* association matrix to *ternary* adjaceny matrix
3. Plotting the figures


### Step 1: Running the Inference Algorithms
#### Feed-forard networks
For the feed-forward case, open up a Terminal (in Mac OS/Linux) or Command Prompt (in Windows), navigate to the `Codes/` folder and type the following commands.

To run *Stochastic NeuInf* type

    $ python Inference_Tomography_FF.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 3

To run *Cross Correlogram* type

    $ python Inference_Tomography_FF.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -M 4
    
    
For GLM, we have to switch to *MATLAB*, and use the code kindly provided by Prof. Jonathan Pillow [here](http://pillowlab.cps.utexas.edu/code_GLM.html).
We have modified the code to adapt it for our own needs, which is in the `GLM/` folder. To run the code, please open MALTAB, navigate to the `GLM\` folder
and perform the following commands

    >> run ./tools_mexcode/initialize_mexcode
    >> run setpaths
    >> T = Infer_GLM(2,[60,12],[15,3],0.2,0.3,'N')
    >> compare_results_GLM(2,[60,12],[15,3],0.2,0.3,'N',T)


#### Recurrent networks
For the recurrent case, open up a Terminal (in Mac OS/Linux) or Command Prompt (in Windows), navigate to the `Codes/` folder and type the following commands.

To run *Stochastic NeuInf* type

    $ python Inference_Tomography.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -M 3

To run *Cross Correlogram* type

    $ python Inference_Tomography.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -M 4
    
For GLM, please open MALTAB, navigate to the `GLM\` folder and perform the following commands

    >> run ./tools_mexcode/initialize_mexcode
    >> run setpaths
    >> T = Infer_GLM(2,[50],[11],0.4,0.1,'N')
    >> compare_results_GLM(2,[50],[11],0.4,0.1,'N',T)


### Step 2: Getteing the Ternary Adjacency Matrix
To transform the analog *association matrices* returned by the previous step, we can simply execute
    
**Feed-forward Networks**

    $ python Transform_to_Ternary.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -B 4 -M 3
    $ python Transform_to_Ternary.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -B 4 -M 4
    $ python Transform_to_Ternary.py -E "60,12" -I "15,3" -L 2 -P "0.0,0.2;0.0,0.0" -D "0.0,9.0;0.0,0.0" -G R -Q 0.3 -B 4 -M 8

**Recurrent Networks**

    $ python Transform_to_Ternary.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 3
    $ python Transform_to_Ternary.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 4
    $ python Transform_to_Ternary.py -E "50" -I "10" -L 1 -P "0.4" -D "10" -Q 0.1 -B 4 -M 8


### Step 3: Plotting the results
After performing the above steps, we can finally reproduce the plots.

    $ python Figure2.py

To perform the evaluations and plot the results, simply execute

    $ python Plot_Results.py [options] 

### Options List
The codes accept a few command line options to identify the specifications of the network and the properties of the neural data. These options are listed below. Note that if an option is not specificed, the *default values* will be used (as specified in the file `Codes/CommonFunctionsdefault_values.py`.

##### General options to specify the network
* `-L xxx`: To specify the number of layers in the network, with `xxx` being an *integer*.
* `-E xxx`: To specify the number of excitatory neurons PER LAYER, as a list given by `xxx`. More specifically, `xxx` has the following format:
  * `-E "n_1,n_2,n_3"`, with `n_i` being the number of excitatory neurons in layer `i`.
* `-I xxx`: To specify the number of inhibitory neurons PER LAYER, as a list given by `xxx`. More specifically, `xxx` has the following format:
  * `-I "m_1,m_2,m_3"`, with `m_i` being the number of inhibitory neurons in layer `i`.
* `-P xxx`: To specify the connection probability from each layer to another layer, as a matrix given by `xxx`. More specifically, `xxx` has the following format:
  * `-P "p_11,p_12;p_21,p_22"`, with `p_ij` being the being the probability of having a connection from layer `i` to `j`. The rows of matrix should be separated from each other with a `;`.
* `-D xxx`: To specify the *maximum* connection delays from each layer to another layer, as a matrix given by `xxx`. More specifically, `xxx` has the following format:
  * `-D "d_11,d_12;d_21,d_22"`, with `d_ij` being the being the maximum connection delays from layer `i` to `j`. The rows of matrix should be separated from each other with a `;`.
* `-Q xxx`: To specify the fraction of stimulated input neurons or probabaility of a neuron being triggered by "outside" traffic, with `xxx` being a *float*.
* `-S xxx`: To specify the number of generated (random) graphs, with `xxx` being an *integer*.
* * `-R xxx`: To specify if the delays are fixed (`R=0`) or random (`R=1`), with `xxx` being an *integer* (it is usally set to *1*, the other option was for earlier versions of the algorithm).
* `-F xxx`: To specify the index of the graphs to start simulations with, with `xxx` being an *integer*.


##### Options for generating data procedure
* `-T xxx`: To specify the duration of the recorded data (in milisecons), with `xxx` being an *integer*.
* `-G xxx`: To specify the method used to generate data, with `xxx` being a *character*.
  * `-G R`: for the general case, where all neurons in the network can be trigerred due to external traffic  
  * `-G F`: for *stimulate-and-observe* scenario
  
##### Options for the inference algorithms
* `-M xxx`: To specify the method use for inference, with `xxx` being an *integer*
  * `-M 3` for *STOCHASTIC NEUINF* (this paper)
  * `-M 4` for Cross Correlogram
* `-K xxx`: To specify the method used to generate data, with `xxx` being a *character*
  * `-K N`: for the general case, where the topology of the network is unknown a priori
  * `-K Y`: for the *topology-aware* scenario (in *fee-forward* networks)
* `-U xxx`: To specify the inverse of the parameter *β* in Stochastic NeuInf algorithm , with `xxx` being an *integer* (the *β* will then be `β = 1/xxx`)
* `-Z xxx`: To specify the (initial) learning rate in Stochastic NeuInf algorithm, *α*, with `xxx` being a *float* 
* `-Y xxx`: To specify if the algorithm should reguralize for sparsity, with `xxx` being an *0/1* [not used in the current version].
* `-X xxx`: To specify the maximum number of iterations inference algorithm is performed, with `xxx` being an *integer* [not used in the current version]

 
##### Options for the digitization algorithms
* `-B xxx`: To specify the ternarification algorithm, with `xxx` being an *integer*.
  * `-B 4` for clustering-based approach (using K-Means)
  * `-B 2` for thresholding-based approach
  * `-B 7` for the conservative approach of only assigning those edges that we are sure about (far from mean values)

##### Options for evaluating performance and plotting the results
* `-O xxx`: To specify the range of recorded durations to evaluate the performance upon, as a list given by `xxx`. More specifically, `xxx` has the following format:
  * `-O "T_1,T_2,T_3"`, where `T_i` (an *integer* in miliseconds) is the duration of recording in session `i`.
* `-f xxx`: To specify the type of plots that should be displayed, as a list given by `xxx`. More specifically, `xxx` has the following format:
  * `-f "F_1,F_2,F_3"`, where `F_i` (a *character*) is the plot type. The flags can be
    * B: for displaying the avaerage value of beliefs
    * P: for displaying precision and recall
    * S: for displaying scatter plots
    * W: to show a sample of inferred graphs
  

### Dependencies
* A working distribution of [Python 2.7](https://www.python.org/downloads/).
* The code relies heavily on [Numpy](http://www.numpy.org/),
  [Scipy](http://www.scipy.org/), and [matplotlib](http://matplotlib.org).
* To generate neural data (using the `Generate_Neural_Data.py`), the code uses [Brian simulator](http://briansimulator.org/).


### The codes have been successfully tested on
* Mac OS 10.9.5, with Python 2.7
* Linux Ubuntu 12.04, with Python 2.7
* Linux Red Hat Enterprise Server 6.5, with Python 2.7
* Microsoft Windows 8, with Python 2.7

Licence
===================
Copyright (C) 2013 Laboratory of Audiovisual Communications (LCAV),
Ecole Polytechnique Federale de Lausanne (EPFL),
CH-1015 Lausanne, Switzerland.

<a rel="license" href="https://en.wikipedia.org/wiki/GNU_General_Public_License"><img alt="GNU General Public License" style="border-width:0" src="http://rr.epfl.ch/img/GNU.png" /></a><br />

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for more details (enclosed in the file GPL).


Credits and Contacts
===================
Amin Karbasi, Amir Hesam Salvati and Martin Vetterli
Laboratory for Audiovisual Communications ([LCAV](http://lcav.epfl.ch)) at 
[EPFL](http://www.epfl.ch).
<img src="http://lcav.epfl.ch/files/content/sites/lcav/files/images/Home/LCAV_anim_200.gif">

#### Contacts and queries
[Amir Hesam Salavati](mailto: saloot[at]gmail[dot]com) <br>







