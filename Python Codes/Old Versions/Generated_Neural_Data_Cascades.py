#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 320
n_inh_default = 80
connection_prob_default = 0.15
frac_input_neurons_default = 0.23
no_cascades_default = 5000
ensemble_size_default = 1
file_name_base_data_default = "./Data"
#==============================================================================

#================================INSTRUCTIONS==================================
help_message = "\n"
help_message = help_message + "\n"
help_message = help_message + "###################################INSTRUCTIONS################################\n"
help_message = help_message + "Here is how to use the code: you have to specify the option flag and"
help_message = help_message + "the quantity right afterwards.\nExample: -E 100 for setting a network with 100 excitatory neurons. "
help_message = help_message + "The full list of options are as follows:\n"
help_message = help_message + "-E xxx: To specify the number of excitatory neurons. Default value = %s.\n" %str(n_exc_default)
help_message = help_message + "-I xxx: To specify the number of inhibitory neurons. Default value = %s.\n" %str(n_inh_default)
help_message = help_message + "-P xxx: To specify the probabaility of having a connection between two neurons. Default value = %s.\n" %str(connection_prob_default)
help_message = help_message + "-Q xxx: To specify the fraction of stimulated input neurons. Default value = %s.\n" %str(frac_input_neurons_default)
help_message = help_message + "-T xxx: To specify the number of considered cascades. Default value = %s.\n" %str(no_cascades_default)
help_message = help_message + "-S xxx: To specify the number of generated random graphs. Default value = %s.\n" %str(ensemble_size_default)
help_message = help_message + "-A xxx: To specify the folder that stores the generated data. Default value = %s. \n" %str(file_name_base_data_default)
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================


#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import time
import numpy as np
import os
import sys,getopt,os
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time.time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:")
if (input_opts):
    for opt, arg in input_opts:
        if opt == '-E':
            n_exc = int(arg)                                    # The number of excitatory neurons in the output layer
        elif opt == '-I':
                n_inh = int(arg)                                    # The number of inhibitory neurons in the output layer
        elif opt == '-P':
            connection_prob = float(arg)                        # The probability of having a link from the input to output neurons in the second layer
        elif opt == '-Q':
            frac_input_neurons = float(arg)                     # Fraction of neurons in the input layer that will be excited by a stimulus
        elif opt == '-T':
            no_cascades = int(arg)                              # Number of times we inject stimulus to the network
        elif opt == '-S':
            ensemble_size = int(arg)                            # The number of random networks that will be generated
        elif opt == '-B':
            binary_mode = int(arg)                              # Defines the method to transform the graph to binary. "1" for threshold base and "2" for sparsity based
        elif opt == '-h':
            print(help_message)
            sys.exit()
else:
    print('Code will be executed using default values')
 
#==============================================================================


#================================INITIALIZATIONS===============================

#------------Set the Default Values if Variables are not Defines---------------
if 'n_exc' not in locals():
    n_exc = n_exc_default
    print('ATTENTION: The default value of %s for n_exc is considered.\n' %str(n_exc))

if 'n_inh' not in locals():
    n_inh = n_inh_default
    print('ATTENTION: The default value of %s for n_inh is considered.\n' %str(n_inh))

if 'connection_prob' not in locals():    
    connection_prob = connection_prob_default
    print('ATTENTION: The default value of %s for connection_prob is considered.\n' %str(connection_prob))

if 'frac_input_neurons' not in locals():
    frac_input_neurons = frac_input_neurons_default
    print('ATTENTION: The default value of %s for frac_input_neurons is considered.\n' %str(frac_input_neurons))

if 'no_cascades' not in locals():        
    no_cascades = no_cascades_default
    print('ATTENTION: The default value of %s for no_cascades is considered.\n' %str(no_cascades))

if 'ensemble_size' not in locals():            
    ensemble_size = ensemble_size_default
    print('ATTENTION: The default value of %s for ensemble_size is considered.\n' %str(ensemble_size))
    
if 'file_name_base_data' not in locals():
    file_name_base_data = file_name_base_data_default;
    print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_data))
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if NEcessary---------------
if not os.path.isdir(file_name_base_data):
    os.makedirs(file_name_base_data)
    temp = file_name_base_data + '/Graphs'
    os.makedirs(temp)
    temp = file_name_base_data + '/Spikes'
    os.makedirs(temp)
#------------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
n = n_exc + n_inh                       # Total number of neurons in the output layer

synapse_delay = 0.0                     # The synaptic delay of ALL links in ms

input_stimulus_freq = 20000               # The frequency of spikes by the neurons in the input layer (in Hz)


no_samples_per_cascade = 3.0              # Number of samples that will be recorded
running_period = (no_samples_per_cascade/10.0)  # Total running time in seconds
#------------------------------------------------------------------------------

#----------------------------------Neural Model--------------------------------
tau=10*ms
tau_e=2*ms # AMPA synapse
eqs='''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''

def myrates(t):
    rates=zeros(n)*Hz    
    if t < 0.1 * ms:
        input_index = floor(n*rand(round(n*frac_input_neurons)))
        input_index = input_index.astype(int)
        rates[input_index]=ones(round(n*frac_input_neurons))*input_stimulus_freq *Hz
    return rates
#------------------------------------------------------------------------------

#==============================================================================




for ensemble_count in range(0,ensemble_size):

#============================GENERATE THE NETWORK==============================
    
    #----------------------Initialize the Main Layer---------------------------
    main_network=NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)

    Pe = main_network.subgroup(n_exc)
    Pi = main_network.subgroup(n_inh)
    Ce = Connection(Pe, main_network, weight=1*mV, sparseness=connection_prob,delay = synapse_delay*ms)
    Ci = Connection(Pi, main_network, weight=-1*mV, sparseness=connection_prob,delay = synapse_delay*ms)
        
        
    #--------------------------------------------------------------------------
    
    
    #--------------Transform Connections to Weighted Matrices------------------
    #Wf = input_connections.W.todense()
    We = Ce.W.todense()
    Wi = Ci.W.todense()
    #--------------------------------------------------------------------------
    
    
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "n_exc_%s" %str(int(n_exc))
    file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
    file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
    file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str(synapse_delay)
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #--------------------------------------------------------------------------
    
    #----------------Save Connectivity Matrices to the File--------------------
    #np.savetxt("/Hesam/Academic/Network Tomography/Data/Graphs/Wf_cascades_%s.txt" %file_name_ending ,Wf,'%1.4f',delimiter='\t',newline='\n')
    file_name = file_name_base_data + "/Graphs/We_cascades_%s.txt" %file_name_ending
    np.savetxt(file_name,We,'%1.4f',delimiter='\t',newline='\n')
    file_name = file_name_base_data + "/Graphs/Wi_cascades_%s.txt" %file_name_ending
    np.savetxt(file_name,Wi,'%1.4f',delimiter='\t',newline='\n')
    #--------------------------------------------------------------------------
 
    #-------------------Run the Network and Record Spikes----------------------
    file_name = file_name_base_data + "/Spikes/S_times_cascades_%s.txt" %file_name_ending
    S_time_file = open(file_name,'w')
    for cascade_count in range(0,no_cascades):
        
        #-------------------Initialize the Input Stimulus----------------------        
        inputer_dummy_layer=PoissonGroup(n,myrates)
        input_connections=Connection(inputer_dummy_layer,main_network,weight=lambda i,j:(1-abs(sign(i-j))),delay = 0*ms)
        #----------------------------------------------------------------------
        
        
        M_l1 = SpikeMonitor(inputer_dummy_layer)
        M_l2 = SpikeMonitor(main_network)
        M_l1.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0
        M_l2.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0
        run(running_period * ms)        
    
        print M_l1.nspikes, "spikes in input layer"
        print M_l2.nspikes, "spikes in output layer"
        #----------------------------------------------------------------------
    
        #--------------------Save Spike Times to the File----------------------        
        
        
        SS = M_l2.spiketimes
        for l in range(0,len(SS)):
            item = SS[l]
            if (len(item) == 0):
                S_time_file.write("0 \t")
            else:
                for j in range(0,len(item)):
                    if (item[j]<0.00015):
                        S_time_file.write("%f \t" % item[j])
                    elif (item[j]>0.00015):
                        S_time_file.write("%f \t" % item[j])
                        break                
            S_time_file.write("-1 \n")
    
        if (cascade_count<no_cascades-1):
            S_time_file.write("-2 \n")
        #----------------------------------------------------------------------
            reinit()
            Reset(resetvalue=0.0 * volt, state=0)
            M_l1.reinit()                               # Reset the spikemonitor so that for the next random network, everything starts from t=0
            M_l2.reinit()                               # Reset the spikemonitor so that for the next random network, everything starts from t=0
            M_l1.source.reinit()
            M_l2.source.reinit()
            M_l1.source.reset()            
            M_l2.source.reset()
    S_time_file.close()
#==============================================================================