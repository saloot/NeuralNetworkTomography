#=======================DEFAULT VALUES FOR THE VARIABLES=======================
FRAC_STIMULATED_NEURONS_DEFAULT = 0.4
NO_STIMUL_ROUNDS_DEFAULT = 1000
ENSEMBLE_SIZE_DEFAULT = 5
FILE_NAME_BASE_DATA_DEFAULT = "./Data"
ENSEMBLE_COUNT_INIT_DEFAULT = 0
GENERATE_DATA_MODE_DEFAULT = 'R'
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

#================================INITIALIZATIONS===============================
n_exc_array = None; n_inh_array= None; connection_prob_matrix = None
random_delay_flag = None; no_layers = None; delay_max_matrix = None

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:G:")
if (input_opts):
    for opt, arg in input_opts:        
        if opt == '-Q':
            frac_stimulated_neurons = float(arg)                # Fraction of neurons in the input layer that will be excited by a stimulus
        elif opt == '-T':
            no_stimul_rounds = int(arg)                         # Number of times we inject stimulus to the network
        elif opt == '-S':
            ensemble_size = int(arg)                            # The number of random networks that will be generated                
        elif opt == '-A':
            file_name_base_data = str(arg)                      # The folder to store results
        elif opt == '-F':
            ensemble_count_init = int(arg)                      # The ensemble to start simulations from
        elif opt == '-G':
            generate_data_mode = str(arg)                       # The data generating method            
        elif opt == '-h':
            print(help_message)
            sys.exit()
else:
    print('Code will be executed using default values')
#==============================================================================


#================================INITIALIZATIONS===============================

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
    file_name_base_data = FILE_NAME_BASE_DATA_DEFAULT
    print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_data))

if 'ensemble_count_init' not in locals():
    ensemble_count_init = ENSEMBLE_COUNT_INIT_DEFAULT
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))
    
if 'generate_data_mode' not in locals():
    generate_data_mode = GENERATE_DATA_MODE_DEFAULT
    print('ATTENTION: The default value of %s for generate_data_mode is considered.\n' %str(generate_data_mode))
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

#Network = NeuralNet(no_layers,n_exc_array,n_inh_array,connection_prob_matrix,delay_max_matrix,random_delay_flag,neural_model_eq,'inline','','')
Network = NeuralNet(None,None,None,None,None,None,None, 'command_line',input_opts,args)

no_samples_per_cascade = max(3.0,25*Network.no_layers*np.max(Network.delay_max_matrix)) # Number of samples that will be recorded
if generate_data_mode == 'F':
    running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
else:    
    running_period = (no_stimul_rounds*no_samples_per_cascade/10.0)  # Total running time in mili seconds
    no_stimul_rounds = 1
#==============================================================================




for ensemble_count in range(ensemble_count_init,ensemble_size):

#============================GENERATE THE NETWORK==============================
        
    #-----------------Create the Weights If They Do Not Exist------------------
    if not Network.read_weights(ensemble_count,file_name_base_data):
        Network.create_weights(ensemble_count,file_name_base_data)
        print "Weight Created"
    #--------------------------------------------------------------------------
    
        
    #-------------------Run the Network and Record Spikes----------------------
    file_name_base = file_name_base_data + "/Spikes/S_times_%s" %Network.file_name_ending + "_q_%s" %str(frac_stimulated_neurons) + '_G_' + generate_data_mode
    
    for l_in in range(0,Network.no_layers):
        file_name = file_name_base + '_l_' + str(l_in) +'.txt'
        S_time_file = open(file_name,'w')
        S_time_file.close()
     
    for cascade_count in range(0,no_stimul_rounds):
        
        #--------------------------Generate Activity---------------------------
        Neural_Connections_Out = generate_neural_activity(Network,running_period,file_name_base,frac_stimulated_neurons,cascade_count,generate_data_mode)
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
