#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian.library.IF import *
from brian.library.synapses import *
import time
import numpy as np 
#==============================================================================


#================================INITIALIZATIONS===============================
n_exc = 320                             # The number of excitatory neurons in the output layer
n_inh = 80                              # The number of inhibitory neurons in the output layer
n = n_exc + n_inh                       # Total number of neurons in the output layer
n_1stlayer = round(2*n)                 # Total number of neurons in the input layer (all excitatory)

FF_flag = 1                             # If 1, we are having only feed forward network, so there will be no recurrent connections in the second layer

synapse_delay = 1                       # The synaptic delay of ALL links in ms


connection_prob = 0.15                  # The probability of having a link from the input to output neurons in the second layer
if (FF_flag==0):
    connection_prob_l2 = 0.15           # The probability of having RECURRENT connections in the output layer


input_stimulus_freq = 800               # The frequency of spikes by the neurons in the input layer (in Hz)
frac_input_neurons = round(n_1stlayer)  # Number of neurons in the input layer that will be excited by a stimulus

no_samples = 10000                      # Number of samples that will be recorded
running_period = int(no_samples/10000)  # Total running time in seconds
ensemble_size = 1                       # The number of random networks that will be generated

#----------------------------------Neural Model--------------------------------
tau=10*ms
tau_e=2*ms # AMPA synapse
eqs='''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''
#------------------------------------------------------------------------------

#==============================================================================




for ensemble_count in range(0,ensemble_size):

#============================GENERATE THE NETWORK==============================
    
    #----------------------Initialize the Input Layer--------------------------
    rates=zeros(n_1stlayer)*Hz
    input_index = floor(n_1stlayer*rand(frac_input_neurons))
    input_index = input_index.astype(int)
    rates[input_index]=ones(frac_input_neurons)*input_stimulus_freq *Hz
    layer1=PoissonGroup(n_1stlayer,rates=rates)
    #--------------------------------------------------------------------------
    
    #----------------------Initialize the Main Layer---------------------------
    layer2=NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)
    #topomap=lambda i,j:exp(-abs(i-j)*.1)*3*mV
    feedforward=Connection(layer1,layer2,sparseness=connection_prob,weight=0.45*mV,delay = synapse_delay*ms)
    
    if (FF_flag==0):
        Pe =layer2.subgroup(n_exc)
        Pi = layer2.subgroup(n_inh)
        Ce = Connection(Pe, layer2, weight=0.45*mV, sparseness=connection_prob_l2,delay = synapse_delay*ms)
        Ci = Connection(Pi, layer2, weight=-0.9*mV, sparseness=connection_prob_l2,delay = synapse_delay*ms)
        
        #lateralmap=lambda i,j:exp(-abs(i-j)*.05)*0.5*mV
        #recurrent=Connection(layer2,layer2,sparseness=connection_prob,weight=lateralmap)
    #--------------------------------------------------------------------------
    
    
    #--------------Transform Connections to Weighted Matrices------------------
    Wf = feedforward.W.todense()
    if (FF_flag==0):
        We = Ce.W.todense()
        Wi = Ci.W.todense()
    #--------------------------------------------------------------------------
    
    
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "n_f_%s" %str(int(n_1stlayer))
    if FF_flag:
        file_name_ending = file_name_ending + "_n_o_%s" %str(int(n))
    else:
        file_name_ending = file_name_ending + "_n_exc_%s" %str(int(n_exc))
        file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))
    
    file_name_ending = file_name_ending + "_n_inp_%s" %str(int(frac_input_neurons))
    file_name_ending = file_name_ending + "_p1_%s" %str(connection_prob)
    if (FF_flag==0):
        file_name_ending = file_name_ending + "_p2_%s" %str(connection_prob_l2)
    
    
    file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str(synapse_delay)
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #--------------------------------------------------------------------------
    
    #----------------Save Connectivity Matrices to the File--------------------
    np.savetxt("/Hesam/Academic/Network Tomography/Data/Graphs/Wf_%s.txt" %file_name_ending ,Wf,'%1.4f',delimiter='\t',newline='\n')
    
    if (FF_flag==0):
        np.savetxt("/Hesam/Academic/Network Tomography/Data/Graphs/We_%s.txt" %file_name_ending,We,'%1.4f',delimiter='\t',newline='\n')
        np.savetxt("/Hesam/Academic/Network Tomography/Data/Graphs/Wi_%s.txt" %file_name_ending,Wi,'%1.4f',delimiter='\t',newline='\n')
    #--------------------------------------------------------------------------
 
    #-------------------Run the Network and Record Spikes----------------------
    M_l1 = SpikeMonitor(layer1)
    M_l2 = SpikeMonitor(layer2)
    run(running_period * second)

    M_l1.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0
    M_l2.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0
    
    print M_l1.nspikes / n_1stlayer, "spikes per neuron in layer 1"
    print M_l2.nspikes / n, "spikes per neuron in layer 2"
    #--------------------------------------------------------------------------
    
    #----------------------Save Spike Times to the File------------------------
    S_time_file_l1 = open("/Hesam/Academic/Network Tomography/Data/Spikes/S_times_l1_%s.txt" %file_name_ending,'w')
    S_time_file_l2 = open("/Hesam/Academic/Network Tomography/Data/Spikes/S_times_l2_%s.txt" %file_name_ending,'w')

    SS = M_l1.spiketimes
    for item in SS.items():
        if len(item)>1:
            itt = item[1]
            for j in range(0,len(itt)-1):
                S_time_file_l1.write("%s \t" % itt[j])
            S_time_file_l1.write("-1 \n")
            
    SS = M_l2.spiketimes
    for item in SS.items():
        if len(item)>1:
            itt = item[1]
            for j in range(0,len(itt)-1):
                S_time_file_l2.write("%s \t" % itt[j])
            S_time_file_l2.write("-1 \n")
    #--------------------------------------------------------------------------

    M_l1.reinit()                               # Reset the spikemonitor so that for the next random network, everything starts from t=0
    M_l2.reinit()                               # Reset the spikemonitor so that for the next random network, everything starts from t=0