Introduction
===================
This repository contains the code for the paper [Inverse Neural](http://rr.epfl.ch/paper/KSV2015). In what follows, we describe how to execute the code, how different options work and, more importantly, how to reproduce the results in the paper.

Executing the Code
===================
The codes are written in *Python*. The required libraries and dependencies are listed further down. Here, we discuss how to run the code. In principle, there are several to get the results shown in the paper, one of which is running the algorithm itself and the others involve generating data and evaluating the performance of the algorithm. The steps are listed below:

1. Generating data [★]
2. Running the algorithm (NeuInf)
3. Transforming the resulting *analog* association matrix to *ternary* adjaceny matrix [★★]
4. Evaluating the performance

★: if you have your own data, there is no need to the run the first step. Simply feed your data to the code. However, doing that for the files provided here could be a bit tricky. You could always [contact us](saloot@gmail.com) of course, but it is also worth checking [our other repository](https://github.com/saloot/NeuralNetworkTomography/tree/master/Network%20Tomography%20Toolbox) dedicated to a more general toolbox-like code. 

★★: This step is optional. If done, the code in step 4 will calculate *precision* and *recall* as well.

Let's start explaining each step in more details
### Step 1: **Generating Data**
To generate data, we should use the file `Generate_Neural_Data.py` in the `Codes/` folder. The run the code, first open up a Terminal (in Mac OS/Linux) or Command Prompt (in windows). Then, navigate to the `Codes/` folder and execute the following command

    $ python Generate_Neural_Data.py [options] 

The code generate spikes times corresponding to an artificial neural network with characteristics specified by the `options` (the list of options will be explained later). 

##### Data Format
The spiking times will be stored in a `.txt` file, where the information about each spike is stored in two columns, separated by a "tab":

1. The first column indicates the index of the neuron that has fired (an *integer*)
2. The second column shows the spike time, in seconds (a *float*)

An example is shown below:

    10    0.004
    13    0.012
    2     0.025

The above example is showing that neurons *10*, *13* and *2* has fired at times *4*ms, *12*ms and *25*ms, respectively.

### Step 2: Running the Inference Algorithm
Assuming we have generated the data using the previous step, all we have to do is to type the folloding commands in the Terminal (within the `Codes/` folder) with the same list of options. For general **recurrent** networks, we run

    $ python Inference_Tomography.py [options] 
    
and for **feedforward** networks, we run

    $ python Inference_Tomography_FF.py [options] 

The results of the inference algorithm will be saved in the `Results/Inferred_Graphs/` folder as analog *association matrices*.

### Step 3: Getteing the Ternary Adjacency Matrix
To transform the analog *association matrices* returned by the previous step, we can simply execute
    
    $ python Transform_to_Ternary.py [options] 

The results will be saved in the `Results/Inferred_Graphs/`. The codes uses a variety of *standard* methods to perform the digitization task (including [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) and simple thresholding techniques).

### Step 4: Evaluating the Performance and Plotting the Resuls
Finally, we can evaluate the performance of the algorithm according to different criteria
1. Quality of association matrix: we calculate the *average* of the values returned by the algorithm (i.e. it's "beliefs") for excitatiroy, inhibitory and non-existen ("void") connections. In principle, we expect the average for excitatory connections be higher than void and then higher than the inhibitory connections.
2. Quality of adjacency matrix: after transforming the association matrix into the a ternary digital one, we can calculate *[precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)* to find out how accurately the algorithm is capable of identifying excitatory and inhibitory connections.
3. Effect of sparisty on the performance: we can assess the performance of the algorithms in different scenarios and see how a sparser network or sparser data improves/deteriorates the performance.

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
  
##### Options for the inference and digitization algorithm
* `-M xxx`: To specify the method use for inference, with `xxx` being an *integer*.
  * `-M 3` for *STOCHASTIC NEUINF* (this paper)
  * `-M 4` for Cross Correlogram
* `-K xxx`: To specify the method used to generate data, with `xxx` being a *character*.
  * `-K N`: for the general case, where the topology of the network is unknown a priori
  * `-K Y`: for the *topology-aware* scenario (in *fee-forward* networks)
* `-U xxx`: To specify if the inverse of the parameter *β* in Stochastic NeuInf algorithm , with `xxx` being an *integer* (the *β* will then be `β = 1/xxx`.)
* `-Y xxx`: To specify if the algorithm should reguralize for sparsity, with `xxx` being an *0/1* [not used in the current version].
* `-X xxx`: To specify the maximum number of iterations inference algorithm is performed, with `xxx` being an *integer* [not used in the current version].
  
* `-B xxx`: To specify the ternarification algorithm, with `xxx` being an *integer*.
  * `-B 4` for clustering-based approach (using K-Means)
  * `-B 2` for thresholding-based approach
  * `-B 7` for the conservative approach of only assigning those edges that we are sure about (far from mean values)

The above options are the more crucial ones. However, the ones below can also help in performing more accurate simulations:


* 



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

Credits and Contacts
===================
