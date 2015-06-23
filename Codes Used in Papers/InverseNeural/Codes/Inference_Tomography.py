#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import sys,getopt,os
import matplotlib.pyplot as plt
import pdb
from copy import deepcopy

from CommonFunctions.auxiliary_functions import read_spikes,combine_weight_matrix,combine_spikes_matrix,generate_file_name
from CommonFunctions.auxiliary_functions_inference import *
from CommonFunctions.Neurons_and_Networks import *

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:X:Y:C:V:J:")

frac_stimulated_neurons,no_stimul_rounds,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,binary_mode,file_name_base_results,inference_method,sparsity_flag,we_know_location,verify_flag,beta,alpha0,infer_itr_max = parse_commands_inf_algo(input_opts)
#==============================================================================


#================================INITIALIZATIONS===============================

#--------------------------Initialize the Network------------------------------
Network = NeuralNet(None,None,None,None,None,None,None, 'command_line',input_opts,args)
#------------------------------------------------------------------------------    

#---------------------Initialize Simulation Variables--------------------------
no_samples_per_cascade = max(3.0,25*Network.no_layers*np.max(Network.delay_max_matrix)) # Number of samples that will be recorded
running_period = (no_stimul_rounds*no_samples_per_cascade/10.0)  # Total running time in mili seconds
no_stimul_rounds = 1

theta = 0.005                                               # The update threshold of the neurons in the network
sparse_thr0 = 0.0001
t_base = time()
#------------------------------------------------------------------------------

#-------------------------Initialize Inference Parameters----------------------

#...........................SOTCHASTIC NEUINF Approach.........................
if (inference_method == 3) or (inference_method == 2):
    d_window = 8                            # The time-window over which we count spikes from pre-synaptic neurons
    bin_size = 5                            # The bin size for creating a rough version of spike times (in miliseconds)
    inferece_params = [alpha0,sparse_thr0,sparsity_flag,theta,250,d_window,beta,bin_size]
#..............................................................................

#...............................CCF-Based Approach.............................
elif (inference_method == 4):
    d_window = 15
    inferece_params = [d_window]            
#..............................................................................

#------------------------------------------------------------------------------

#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
for ensemble_count in range(ensemble_count_init,ensemble_size):

    t0_ensemble = time()
    
    #--------------------------READ THE NETWORK--------------------------------
    Network.read_weights(ensemble_count,file_name_base_data)
    #--------------------------------------------------------------------------
    
    #--------------------------Read and Sort Spikes----------------------------
    file_name_base = file_name_base_data + "/Spikes/S_times_%s" %Network.file_name_ending + "_q_%s" %str(frac_stimulated_neurons) + '_G_' + generate_data_mode
    Neural_Spikes,T_max = read_spikes(file_name_base,Network.no_layers,Network.n_exc_array,Network.n_inh_array,['u',int(running_period*10)])
    #--------------------------------------------------------------------------
    
    #------Calculate the Range to Assess the Effect of Recording Duration------
    T_max = int(1000*T_max)
    T_step = int(T_max/6.0)
    T_range = range(T_step, T_max+1, T_step)
    #--------------------------------------------------------------------------
    
#==============================================================================

#============================INFER THE CONNECTIONS=============================
       
    #--------------------Read the Whole Connectivity Matrix--------------------
    W,DD_tot = combine_weight_matrix(Network)    
    m,n = W.shape
    #--------------------------------------------------------------------------           
    
    #------------------------Infer the Graph For Each T------------------------
    for T in T_range:
        
        #.....................Assign Inference Parameters......................
        out_spikes_tot,out_spikes_tot_mat,non_stimul_inds = combine_spikes_matrix(Network,T,generate_data_mode,Neural_Spikes)
        
        W_estimated = np.zeros([n,m])
        fixed_entries = np.zeros([n,m])
        #......................................................................
        
    
        #........................Perfrom Inference.........................
        for infer_itr in range(0,infer_itr_max):
                
            #-------------------Perform the Inference Step----------------
            W_inferred_our_tot,cost,Inf_Delays = inference_alg_per_layer(out_spikes_tot_mat,out_spikes_tot_mat,inference_method,inferece_params,W_estimated,0,generate_data_mode)
            #-------------------------------------------------------------
        
        #..............................................................
        
        #...................Save the Belief Matrices...................
        file_name_ending = generate_file_name(Network.file_name_ending,inference_method,we_know_location,pre_synaptic_method,generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,delay_known_flag)
        file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
        np.savetxt(file_name,W_inferred_our_tot,'%1.5f',delimiter='\t')
        #..............................................................
                
                
        itr_T = itr_T + 1
    #----------------------------------------------------------------------
