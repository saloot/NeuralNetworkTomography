#=======================IMPORT THE NECESSARY LIBRARIES=========================
import brian
from brian import *
import numpy as np
import sys,getopt,os

from CommonFunctions.auxiliary_functions_generate import generate_neural_activity,parse_commands_gen_data
from CommonFunctions.Neurons_and_Networks import *

spikequeue.SpikeQueue.reinit
os.system('clear')                                              # Clear the commandline window
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:G:")
frac_stimulated_neurons,no_stimul_rounds,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode = parse_commands_gen_data(input_opts)
#==============================================================================


#================================INITIALIZATIONS===============================

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

#----------------Calculate the Running Period for the Simulation---------------
Network = NeuralNet(None,None,None,None,None,None,None, 'command_line',input_opts,args)

no_samples_per_cascade = max(3.0,25*Network.no_layers*np.max(Network.delay_max_matrix)) # Number of samples that will be recorded
if generate_data_mode == 'F':
    running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
else:    
    running_period = (no_stimul_rounds*no_samples_per_cascade/10.0)  # Total running time in mili seconds
    no_stimul_rounds = 1
#------------------------------------------------------------------------------

#==============================================================================


for ensemble_count in range(ensemble_count_init,ensemble_size):

#============================GENERATE THE NETWORK==============================
        
    #-----------------Create the Weights If They Do Not Exist------------------
    if not Network.read_weights(ensemble_count,file_name_base_data):
        Network.create_weights(ensemble_count,file_name_base_data)
        print "Weights Created"
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
