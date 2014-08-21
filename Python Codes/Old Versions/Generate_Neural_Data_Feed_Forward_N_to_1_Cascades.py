#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.2
frac_input_neurons_default = 0.4
no_cascades_default = 5000
ensemble_size_default = 10
delay_max_default = 1.0
file_name_base_data_default = "./Data/FeedForward"
ensemble_count_init_default = 0
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
help_message = help_message + "-D xxx: To specify the maximum delay for the neural connections in milliseconds. Default value = %s.\n" %str(delay_max_default)
help_message = help_message + "-S xxx: To specify the number of generated random graphs. Default value = %s.\n" %str(ensemble_size_default)
help_message = help_message + "-A xxx: To specify the folder that stores the generated data. Default value = %s. \n" %str(file_name_base_data_default)
help_message = help_message + "-F xxx: To specify the ensemble index to start simulation. Default value = %s. \n" %str(ensemble_count_init_default)
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================


#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import time
import numpy as np
import os
import sys,getopt,os
from auxiliary_functions import *

#os.chdir('C:\Python27')
#os.chdir('/home/salavati/Desktop/Neural_Tomography')
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time.time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:D:A:F:")
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
        elif opt == '-D':
            delay_max = float(arg)                              # The maximum amount of synaptic delay in mili seconds
        elif opt == '-A':
            file_name_base_data = str(arg)                      # The folder to store results
        elif opt == '-F':
            ensemble_count_init = int(arg)                      # The ensemble to start simulations from
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

if 'delay_max' not in locals():
    delay_max = delay_max_default;
    print('ATTENTION: The default value of %s for delay_max is considered.\n' %str(delay_max))

if 'ensemble_count_init' not in locals():
    ensemble_count_init = ensemble_count_init_default;
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))    
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if NEcessary---------------
if not os.path.isdir(file_name_base_data):
    os.makedirs(file_name_base_data)

temp = file_name_base_data + '/Graphs'
if not os.path.isdir(temp):    
    os.makedirs(temp)

temp = file_name_base_data + '/Spikes'
if not os.path.isdir(temp):        
    os.makedirs(temp)
#------------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
n = n_exc + n_inh                       # Total number of neurons in the output layer

no_samples_per_cascade = max(3.0,25*delay_max) # Number of samples that will be recorded
running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
#------------------------------------------------------------------------------

#----------------------------------Neural Model--------------------------------
tau=10*ms
tau_e=2*ms # AMPA synapse
eqs='''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''
#------------------------------------------------------------------------------

#==============================================================================




for ensemble_count in range(ensemble_count_init,ensemble_size):

#============================GENERATE THE NETWORK==============================
        
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "n_exc_%s" %str(int(n_exc))
    file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
    file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
    #file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #--------------------------------------------------------------------------
    
    file_name = file_name_base_data + "/Graphs/We_FF_n_1_%s.txt" %file_name_ending
    W_flag = 1
    if (os.path.isfile(file_name)):
        We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        We = We.reshape(n_exc,1)        
    else:
        W_flag = 0
    
    file_name = file_name_base_data + "/Graphs/Wi_FF_n_1_%s.txt" %file_name_ending
    if (os.path.isfile(file_name)):        
        Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        Wi = Wi.reshape(n_inh,1)        
    else:
        W_flag = 0
        
    file_name = file_name_base_data + "/Graphs/De_FF_n_1_%s.txt" %file_name_ending
    W_flag = 1
    if (os.path.isfile(file_name)):
        De = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        De = De.reshape(n_exc,1)        
    else:
        W_flag = 0
    
    file_name = file_name_base_data + "/Graphs/Di_FF_n_1_%s.txt" %file_name_ending
    if (os.path.isfile(file_name)):        
        Di = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        Di = Wi.reshape(n_inh,1)        
    else:
        W_flag = 0        
    
    if not W_flag:        
        #----------------------Initialize the Main Layer---------------------------    
        main_network=NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)
        output_neuron = NeuronGroup(1,model=eqs,threshold=10*mV,reset=0*mV)
        
        Pe = main_network.subgroup(n_exc)
        Pi = main_network.subgroup(n_inh)
        h = float(n_exc)/float(2)
        Ce = Connection(Pe, output_neuron, weight=1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * ms * (i /h - 1) ** 2)
        h = float(n_inh)/float(2)
        Ci = Connection(Pi, output_neuron, weight=-1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * ms * (i /h - 1) ** 2)
        #--------------------------------------------------------------------------
    
    
        #--------------Transform Connections to Weighted Matrices------------------
        #Wf = input_connections.W.todense()
        We = Ce.W.todense()
        Wi = Ci.W.todense()
        De = Ce.delay.todense()
        Di = Ci.delay.todense()
        #--------------------------------------------------------------------------    
    
        #----------------Save Connectivity Matrices to the File--------------------
        #np.savetxt("/Hesam/Academic/Network Tomography/Data/Graphs/Wf_cascades_%s.txt" %file_name_ending ,Wf,'%1.4f',delimiter='\t',newline='\n')
        file_name = file_name_base_data + "/Graphs/We_FF_n_1_%s.txt" %file_name_ending
        np.savetxt(file_name,We,'%1.4f',delimiter='\t',newline='\n')
        file_name = file_name_base_data + "/Graphs/Wi_FF_n_1_%s.txt" %file_name_ending
        np.savetxt(file_name,Wi,'%1.4f',delimiter='\t',newline='\n')
        
        file_name = file_name_base_data + "/Graphs/De_FF_n_1_%s.txt" %file_name_ending
        np.savetxt(file_name,De,'%1.4f',delimiter='\t',newline='\n')
        file_name = file_name_base_data + "/Graphs/Di_FF_n_1_%s.txt" %file_name_ending
        np.savetxt(file_name,Di,'%1.4f',delimiter='\t',newline='\n')
        #--------------------------------------------------------------------------
 
    W = np.vstack((We,Wi))
    #-------------------Run the Network and Record Spikes----------------------
    file_name = file_name_base_data + "/Spikes/S_times_FF_n_1_cascades_%s.txt" %file_name_ending
    S_time_file = open(file_name,'w')
    file_name = file_name_base_data + "/Spikes/S_times_FF_n_1_out_cascades_%s.txt" %file_name_ending
    S_time_file_out = open(file_name,'w')
    for cascade_count in range(0,no_cascades):
        
        #--------------------------Generate Activity---------------------------
        WWe,WWi,DDe,DDi = generate_neural_activity(n_exc,n_inh,running_period,We,Wi,De,Di,S_time_file,S_time_file_out,eqs,tau,tau_e,delay_max,frac_input_neurons,cascade_count,'F')
        #----------------------------------------------------------------------
        
        #------------------------Check for Any Errors--------------------------
        if norm(WWe-We):
            print('something is wrong with We!')
            break
        if norm(WWi-Wi):
            print('something is wrong with Wi!')
            break
        if norm(DDe-De):
            print('something is wrong with De!')
            break
        if norm(DDi-Di):
            print('something is wrong with Di!')
            break
        #----------------------------------------------------------------------
        
    S_time_file_out.close()
    S_time_file.close()
#==============================================================================
