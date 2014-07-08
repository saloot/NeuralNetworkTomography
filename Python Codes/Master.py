#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.2
frac_input_neurons_default = 0.3
no_cascades_default = 8000
ensemble_size_default = 10
delay_max_default = 1.0
ensemble_count_init_default = 0
network_type_default ='F'
OS_default ='W'
folder_codes_default = './' #'C:\Python27'
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
help_message = help_message + "-A xxx: To specify the folder that stores the generated data. Default value = %s. \n" %str(folder_codes_default)
help_message = help_message + "-F xxx: To specify the ensemble index to start simulation. Default value = %s. \n" %str(ensemble_count_init_default)
help_message = help_message + "-N xxx: To specify the network type to simulate. Default value = %s. \n" %str(network_type_default)
help_message = help_message + "-O xxx: The operating system. Default value = %s. \n" %str(OS_default)
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
import subprocess
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time.time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:D:A:F:N:O:")
if (input_opts):
    for opt, arg in input_opts:
        if opt =='-E':
            n_exc = int(arg)                                    # The number of excitatory neurons in the output layer
        elif opt =='-I':
                n_inh = int(arg)                                # The number of inhibitory neurons in the output layer
        elif opt =='-P':
            connection_prob = float(arg)                        # The probability of having a link from the input to output neurons in the second layer
        elif opt =='-Q':
            frac_input_neurons = float(arg)                     # Fraction of neurons in the input layer that will be excited by a stimulus
        elif opt =='-T':
            no_cascades = int(arg)                              # Number of times we inject stimulus to the network
        elif opt =='-S':
            ensemble_size = int(arg)                            # The number of random networks that will be generated
        elif opt =='-B':
            binary_mode = int(arg)                              # Defines the method to transform the graph to binary. "1" for threshold base and "2" for sparsity based
        elif opt =='-D':
            delay_max = float(arg)                              # The maximum amount of synaptic delay in mili seconds
        elif opt =='-A':
            folder_codes = str(arg)                             # The folder to store results
        elif opt =='-F':
            ensemble_count_init = int(arg)                      # The ensemble to start simulations from
        elif opt =='-N':
            network_type = str(arg)                             # The type of network to simulate
        elif opt =='-O':
            OS_type = str(arg)                                  # The operating system
        elif opt =='-h':
            print(help_message)
            sys.exit()
else:
    print('Code will be executed using default values')
 
#==============================================================================


#================================INITIALIZATIONS===============================

#------------Set the Default Values if Variables are not Defines---------------
if'n_exc' not in locals():
    n_exc = n_exc_default
    print('ATTENTION: The default value of %s for n_exc is considered.\n' %str(n_exc))

if'n_inh' not in locals():
    n_inh = n_inh_default
    print('ATTENTION: The default value of %s for n_inh is considered.\n' %str(n_inh))

if'connection_prob' not in locals():    
    connection_prob = connection_prob_default
    print('ATTENTION: The default value of %s for connection_prob is considered.\n' %str(connection_prob))

if'frac_input_neurons' not in locals():
    frac_input_neurons = frac_input_neurons_default
    print('ATTENTION: The default value of %s for frac_input_neurons is considered.\n' %str(frac_input_neurons))

if'no_cascades' not in locals():        
    no_cascades = no_cascades_default
    print('ATTENTION: The default value of %s for no_cascades is considered.\n' %str(no_cascades))

if'ensemble_size' not in locals():            
    ensemble_size = ensemble_size_default
    print('ATTENTION: The default value of %s for ensemble_size is considered.\n' %str(ensemble_size))
    
if'folder_codes' not in locals():
    folder_codes = folder_codes_default;
    print('ATTENTION: The default value of %s for folder_codes is considered.\n' %str(folder_codes))

if'delay_max' not in locals():
    delay_max = delay_max_default;
    print('ATTENTION: The default value of %s for delay_max is considered.\n' %str(delay_max))

if'ensemble_count_init' not in locals():
    ensemble_count_init = ensemble_count_init_default;
    print('ATTENTION: The default value of %s for str(ensemble_count_init) is considered.\n' %str(ensemble_count_init))

if'network_type' not in locals():
    network_type = network_type_default;
    print('ATTENTION: The default value of %s for network_type is considered.\n' %str(network_type))
    
#------------------------------------------------------------------------------

#==============================================================================

#==========================EXECUTE THE COMMANDS================================
if (network_type =='F'):
    subprocess.call(['python',folder_codes +'Generate_Neural_Data_Feed_Forward_N_to_1_Cascades.py','-E', str(n_exc),'-I',
                     str(n_inh),'-P', str(connection_prob),'-Q', str(frac_input_neurons),'-T',str(no_cascades),
                    '-S', str(ensemble_size),'-D',str(delay_max),'-F',str(ensemble_count_init)])
    
    subprocess.call(['python',folder_codes +'Inference_FF_n_to_1_Cascades.py','-E', str(n_exc),'-I',
                     str(n_inh),'-P', str(connection_prob),'-Q', str(frac_input_neurons),'-T',str(no_cascades),
                    '-S', str(ensemble_size),'-D',str(delay_max),'-F',str(ensemble_count_init)])
    
    subprocess.call(['python',folder_codes +'Transform_to_Binary.py','-E', str(n_exc),'-I',
                     str(n_inh),'-P', str(connection_prob),'-Q', str(frac_input_neurons),'-T',str(no_cascades),
                    '-S', str(ensemble_size),'-D',str(delay_max),'-F',str(ensemble_count_init),'-N',str(network_type)])
        
elif (network_type =='R'):
    subprocess.call(['python',folder_codes +'Generate_Neural_Data_Delayed_Cascades.py','-E', str(n_exc),'-I',
                     str(n_inh),'-P', str(connection_prob),'-Q', str(frac_input_neurons),'-T',str(no_cascades),
                    '-S', str(ensemble_size),'-D',str(delay_max),'-F',str(ensemble_count_init)])
    
    subprocess.call(['python',folder_codes +'Inference_Cascades_Delayed.py','-E', str(n_exc),'-I',
                     str(n_inh),'-P', str(connection_prob),'-Q', str(frac_input_neurons),'-T',str(no_cascades),
                    '-S', str(ensemble_size),'-D',str(delay_max),'-F',str(ensemble_count_init)])
    
    subprocess.call(['python',folder_codes +'Transform_to_Binary.py','-E', str(n_exc),'-I',
                     str(n_inh),'-P', str(connection_prob),'-Q', str(frac_input_neurons),'-T',str(no_cascades),
                    '-S', str(ensemble_size),'-D',str(delay_max),'-F',str(ensemble_count_init),'-N',str(network_type)])
else:
    print 'Invalid network type'
    

#==============================================================================