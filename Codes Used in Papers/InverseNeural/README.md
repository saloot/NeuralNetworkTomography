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


### Options List

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
