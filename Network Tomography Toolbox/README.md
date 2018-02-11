Introduction
===================
This repository contains the code for the neural network tomography toolbox. In what follows, we describe how to execute the code, how different options work and, more importantly, how to reproduce the results in some of the related papers.

Related Publications
###
* [Inverse Neural](http://rr.epfl.ch/paper/KSV2015)


Executing the Code
===================
We assume that we have a dataset of spike times and would like to identify the underlying neural graph using the algorithms in this toolbox. We start by explaining the proper format of the file containing the spike times and then show you how to apply the inference algorithm, save the results and interpret them. The codes are written in *Python* (Python 2.7 to be more specific). The required libraries and dependencies are listed further down. Here, we discuss how to run the code. The neceary steps are listed below:

1. Running the algorithm (NeuInf)
2. Transforming the resulting *analog* association matrix to *ternary* adjaceny matrix [★★]
3. Evaluating the performance


##### Data Format
Before we continue, let us discuss the format of the file that contains the spiking data. The spiking times should be stored in a `.txt` file, where the information about each spike is stored in two columns, separated by a "tab":

1. The first column indicates the index of the neuron that has fired (an *integer*)
2. The second column shows the spike time, in seconds (a *float*)

An example is shown below:

    10    0.004
    13    0.012
    2     0.025

The above example is showing that neurons *10*, *13* and *2* has fired at times *4*ms, *12*ms and *25*ms, respectively.

### Step 1: Running the Inference Algorithm
Assuming we have the data file with the format discussed in the previous step, all we have to do is to type the folloding commands in the Terminal (within the `Codes/` folder) with a list of options that determine the inference algorithm and its parameters:

    $ python Inference_Tomography.py [options] 

The results of the inference algorithm will be saved in the `Results/Inferred_Graphs/` folder as analog *association matrices*.

### Step 2: Getteing the Ternary Adjacency Matrix
To transform the analog *association matrices* returned by the previous step, we can simply execute
    
    $ python Transform_to_Ternary.py [options] 

The results will be saved in the `Results/Inferred_Graphs/`. The codes uses a variety of *standard* methods to perform the digitization task (including [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) and simple thresholding techniques).

### Step 3: Evaluating the Performance and Plotting the Resuls
Finally, we can evaluate the performance of the algorithm according to different criteria

1. Quality of association matrix: we calculate the *average* of the values returned by the algorithm (i.e. it's "beliefs") for excitatiroy, inhibitory and non-existen ("void") connections. In principle, we expect the average for excitatory connections be higher than void and then higher than the inhibitory connections.
2. Quality of adjacency matrix: after transforming the association matrix into the a ternary digital one, we can calculate *[precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)* to find out how accurately the algorithm is capable of identifying excitatory and inhibitory connections.
3. Effect of sparisty on the performance: we can assess the performance of the algorithms in different scenarios and see how a sparser network or sparser data improves/deteriorates the performance.

To perform the evaluations and plot the results, simply execute

    $ python Plot_Results.py [options] 


### Options List
The codes accept a few command line options to identify the specifications of the network and the properties of the neural data. These options are listed below. Note that if an option is not specificed, the *default values* will be used (as specified in the file `Codes/CommonFunctionsdefault_values.py`.

##### General options to specify the network
* `-T xxx`: To specify the number of rcorded samples or, equivalently, the duration of the recorded data (in milisecons), with `xxx` being an *integer*.
* `-A xxx`: To compelte path to the file that contains the recordings, with `xxx` being a *string*.
* `-N xxx`: To specify the number of observed neurons in the recorded data, with `xxx` being an *integer*.
* `-M xxx`: To specify the inference algorithm, with `xxx` being an *integer*. The acceptable values are listed below, with the default value being *3*:
* * 1: for Stochastic Dual Coordinate Descent (SDCS)
* * 2: for Sketched SDCS
* * 3: Perceptron
* * 4: Sketched Perceptron
* * 5: SDCS with Double Hinge Loss
* * 6: Perceptron with Double Hinge Loss
* * 7: [experimental] Perceptron with Cross Entropy Loss

* `-o xxx`: To specify the range of target neurons to execute the algorithm for. More precisely, the algorithm will find out the incoming connections to the neurons specified in this range. `xxx` has the following format:
  * `-o "n_1,n_2"`, where the algorithm find the connections for neurons in the range of `n_1,...,n_2`, with n_1 and n_2 both being an *integer*.

* `-Q xxx`: To specify the number of cpu cores to use for simulaitons, with `xxx` being an *integer*. The default value 8, if the number of available cores i more than 8. Otherwise it i qual to the number of availble cores.
* `-L xxx`: To specify the type of neural kernel, i.e. the decay function in LIF neurons, with `xxx` being either 'E', for single exponential decay or 'D' for a double exponential decay function. The default value is 'E'.
* `-H xxx`: To specify the number of hidden neurons in the data, with `xxx` being an *integer*. If not 0, the algorithm randomly omits `xxx` neurons from data to count them as hidden and then perform the algorithm to identify the performance in presence of unobserved neurons.
* `-Z xxx`: To specify the (initial) learning rate in Stochastic NeuInf algorithm, *α*, with `xxx` being a *float* 
* `-Y xxx`: To specify the penalty coefficient to reguralize for sparsity, with `xxx` being a *float*.
* `-X xxx`: To specify the maximum number of iterations inference algorithm is performed, with `xxx` being an *integer* [not used in the current version]
* `-U xxx`: To specify the inverse of the parameter *β* in Stochastic NeuInf algorithm , with `xxx` being an *integer* (the *β* will then be `β = 1/xxx`)
* `-J xxx`: To specify the probability of choosing samples from the firing instances, with `xxx` being an *integer*
* `-S xxx`: To specify the block size, i.e. the number of firing samples to loa into RAM for running each batch of algorithm, with `xxx` being an *integer*.
* `-S xxx`: To specify the block size, i.e. the number of firing samples to loa into RAM for running each batch of algorithm, with `xxx` being an *integer*.

 
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







