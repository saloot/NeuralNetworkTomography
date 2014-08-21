#=======================DEFAULT VALUES FOR THE VARIABLES=======================
FRAC_STIMULATED_NEURONS_DEFAULT = 0.4
NO_STIMUL_ROUNDS_DEFAULT = 2000
ENSEMBLE_SIZE_DEFAULT = 1
FILE_NAME_BASE_DATA_DEFAULT = "./Data"
ENSEMBLE_COUNT_INIT_DEFAULT = 0
#==============================================================================


#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import numpy as np
import os
import sys,getopt,os

import auxiliary_functions
reload(auxiliary_functions)
from auxiliary_functions import generate_neural_activity

import Neurons_and_Networks
reload(Neurons_and_Networks)
from Neurons_and_Networks import NeuralNet
from Neurons_and_Networks import *

import brian
reload(brian)
from brian import *
spikequeue.SpikeQueue.reinit
#os.chdir('C:\Python27')
#os.chdir('/home/salavati/Desktop/Neural_Tomography')
#==============================================================================

#================================INSTRUCTIONS==================================
help_message = "\n"
help_message = help_message + "\n"
help_message = help_message + "###################################INSTRUCTIONS################################\n"
help_message = help_message + "Here is how to use the code: you have to specify the option flag and"
help_message = help_message + "the quantity right afterwards.\nExample: -E 100 for setting a network with 100 excitatory neurons. "
help_message = help_message + "The full list of options are as follows:\n"
help_message = help_message + "-E xxx: To specify the number of excitatory neurons PER LAYER (as a list). Default value = '%s'.\n" %str(N_EXC_ARRAY_DEFAULT)
help_message = help_message + "-I xxx: To specify the number of inhibitory neurons. Default value = %s.\n" %str(N_INH_ARRAY_DEFAULT)
help_message = help_message + "-P xxx: To specify the probabaility of having a connection between two neurons. Default value = %s.\n" %str(DELAY_MAX_MATRIX_DEFAULT)
help_message = help_message + "-Q xxx: To specify the fraction of stimulated input neurons. Default value = %s.\n" %str(FRAC_STIMULATED_NEURONS_DEFAULT)
help_message = help_message + "-T xxx: To specify the number of considered cascades. Default value = %s.\n" %str(NO_STIMUL_ROUNDS_DEFAULT)
help_message = help_message + "-D xxx: To specify the maximum delay for the neural connections in milliseconds. Default value = %s.\n" %str(DELAY_MAX_MATRIX_DEFAULT)
help_message = help_message + "-S xxx: To specify the number of generated random graphs. Default value = %s.\n" %str(ENSEMBLE_SIZE_DEFAULT)
help_message = help_message + "-A xxx: To specify the folder that stores the generated data. Default value = %s. \n" %str(FILE_NAME_BASE_DATA_DEFAULT)
help_message = help_message + "-F xxx: To specify the ensemble index to start simulation. Default value = %s. \n" %str(ENSEMBLE_COUNT_INIT_DEFAULT)
help_message = help_message + "-L xxx: To specify the number of layers in the network. Default value = %s. \n" %str(NO_LAYERS_DEFAULT)
help_message = help_message + "-R xxx: To specify if the delays are fixed (R=0) or random (R=1). Default value = %s. \n" %str(RANDOM_DELAY_FLAG_DEFAULT)
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================


#================================INITIALIZATIONS===============================
n_exc_array = None; n_inh_array= None; connection_prob_matrix = None
random_delay_flag = None; no_layers = None; delay_max_matrix = None

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:")
if (input_opts):
    for opt, arg in input_opts:
        if opt == '-E':
            n_exc_array = np.matrix(str(arg))                   # The number of excitatory neurons in each layer
        elif opt == '-I':
            n_inh_array = np.matrix(str(arg))                   # The number of excitatory neurons in each layer            
        elif opt == '-P':
            connection_prob_matrix = np.matrix(str(arg))        # The probability of having a link from each layer to the other. Separate the rows with a ";"
        elif opt == '-Q':
            frac_stimulated_neurons = float(arg)                # Fraction of neurons in the input layer that will be excited by a stimulus
        elif opt == '-T':
            no_stimul_rounds = int(arg)                         # Number of times we inject stimulus to the network
        elif opt == '-S':
            ensemble_size = int(arg)                            # The number of random networks that will be generated        
        elif opt == '-D':
            delay_max_matrix = np.matrix(str(arg))              # The maximum amount of synaptic delay in mili seconds
        elif opt == '-A':
            file_name_base_data = str(arg)                      # The folder to store results
        elif opt == '-F':
            ensemble_count_init = int(arg)                      # The ensemble to start simulations from
        elif opt == '-L':
            no_layers = int(arg)                                # The number of layers in the network
        elif opt == '-R':
            random_delay_flag = int(arg)                        # The ensemble to start simulations from            
        elif opt == '-h':
            print(help_message)
            sys.exit()
else:
    print('Code will be executed using default values')
 
#==============================================================================


#================================INITIALIZATIONS===============================

no_layers = 4

#------------Set the Default Values if Variables are not Defines---------------
if 'frac_stimulated_neurons' not in locals():
    frac_stimulated_neurons = FRAC_STIMULATED_NEURONS_DEFAULT
    print('ATTENTION: The default value of %s for frac_stimulated_neurons is considered.\n' %str(frac_stimulated_neurons))

if 'no_stimul_rounds' not in locals():        
    no_stimul_rounds = NO_STIMUL_ROUNDS_DEFAULT
    print('ATTENTION: The default value of %s for no_stimul_rounds is considered.\n' %str(no_stimul_rounds))

if 'ensemble_size' not in locals():            
    ensemble_size = ENSEMBLE_SIZE_DEFAULT
    print('ATTENTION: The default value of %s for ensemble_size is considered.\n' %str(ensemble_size))
    
if 'file_name_base_data' not in locals():
    file_name_base_data = FILE_NAME_BASE_DATA_DEFAULT;
    print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_data))

if 'ensemble_count_init' not in locals():
    ensemble_count_init = ENSEMBLE_COUNT_INIT_DEFAULT;
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if Necessary---------------
if not os.path.isdir(file_name_base_data):
    os.makedirs(file_name_base_data)

temp = file_name_base_data + '/Graphs'
if not os.path.isdir(temp):    
    os.makedirs(temp)

temp = file_name_base_data + '/Spikes'
if not os.path.isdir(temp):        
    os.makedirs(temp)
#------------------------------------------------------------------------------

#----------------------------------Neural Model--------------------------------
global tau,tau_e
tau=10*ms
tau_e=2*ms # AMPA synapse
eqs='''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''

neural_model_eq = list([eqs,tau,tau_e])
#------------------------------------------------------------------------------

import auxiliary_functions
reload(auxiliary_functions)
from auxiliary_functions import generate_neural_activity


import Neurons_and_Networks
reload(Neurons_and_Networks)
from Neurons_and_Networks import NeuralNet
from Neurons_and_Networks import *

network_type = 'F'
Network = NeuralNet(no_layers,n_exc_array,n_inh_array,connection_prob_matrix,delay_max_matrix,random_delay_flag,neural_model_eq)

no_samples_per_cascade = max(3.0,25*Network.no_layers*np.max(Network.delay_max_matrix)) # Number of samples that will be recorded
if network_type == 'F':
    running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
else:
    running_period = (10*no_samples_per_cascade/10.0)  # Total running time in mili seconds
#==============================================================================




for ensemble_count in range(ensemble_count_init,ensemble_size):

#============================GENERATE THE NETWORK==============================
        
    #-----------------Create the Weights If They Do Not Exist------------------
    if not Network.read_weights(ensemble_count,file_name_base_data):
        Network.create_weights(ensemble_count,file_name_base_data)
    #--------------------------------------------------------------------------
    
        
    #-------------------Run the Network and Record Spikes----------------------
    file_name_base = file_name_base_data + "/Spikes/S_times_%s" %Network.file_name_ending + "_q_%s" %str(frac_stimulated_neurons)
    
    for l_in in range(0,Network.no_layers):
        file_name = file_name_base + '_l_' + str(l_in) +'.txt'
        S_time_file = open(file_name,'w')
        S_time_file.close()
     
    for cascade_count in range(0,no_stimul_rounds):
        
        #--------------------------Generate Activity---------------------------
        Neural_Connections_Out = generate_neural_activity(Network,running_period,file_name_base,frac_stimulated_neurons,cascade_count)
        #----------------------------------------------------------------------
        
        #------------------------Check for Any Errors--------------------------
        for l_in in range(0,Network.no_layers):
            for l_out in range(l_in,Network.no_layers):
                ind = str(l_in) + str(l_out)
                
                temp_list1 = Network.Neural_Connections[ind]
                temp_list2 = Neural_Connections_Out[ind]
                
                if norm(temp_list1[0]-temp_list2[0]):
                    print('something is wrong with W!')
                    break
                if norm(temp_list1[1]-temp_list2[1]):
                    print('something is wrong with D!')
                    break
        #----------------------------------------------------------------------
#==============================================================================
