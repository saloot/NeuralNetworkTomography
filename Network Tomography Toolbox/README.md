### Reusing the algorithm for other datasets
In this part, we assume that you have a dataset of spike times and would like to identify the underlying neural graph using the proposed algorithm in this paper. We start by explaining the proper format of the file containing the spike times and then show you how to apply the inference algorithm, save the results and interpret them.

#### Spikes Times File
We assume the spike times are stored in the following ```.txt``` file:  

```./Data/Spikes/S_times.txt```

##### Data Format
The spiking times should be stored in a ```.txt``` file, where the information about each spike is stored in two columns, separated by a "tab":

1. The first column indicates the index of the neuron that has fired (an *integer*)
2. The second column shows the spike time, in seconds (a *float*)

An example is shown below:

    10    0.004
    13    0.012
    2     0.025
