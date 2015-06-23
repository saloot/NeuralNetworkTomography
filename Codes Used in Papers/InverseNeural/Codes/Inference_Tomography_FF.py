#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import sys,getopt,os
import matplotlib.pyplot as plt
import pdb
from copy import deepcopy

from CommonFunctions.auxiliary_functions import read_spikes,combine_weight_matrix,combine_spikes_matrix
from CommonFunctions.auxiliary_functions_inference import *
from CommonFunctions.Neurons_and_Networks import *

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:G:X:Y:K:C:V:J:")

frac_stimulated_neurons,no_stimul_rounds,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,binary_mode,file_name_base_results,inference_method,sparsity_flag,we_know_location,verify_flag,beta,alpha0,infer_itr_max = parse_commands_inf_algo(input_opts)
#==============================================================================


#================================INITIALIZATIONS===============================

#--------------------------Initialize the Network------------------------------
Network = NeuralNet(None,None,None,None,None,None,None, 'command_line',input_opts,args)
#------------------------------------------------------------------------------    

#---------------------Initialize Simulation Variables--------------------------
no_samples_per_cascade = max(3.0,25*Network.no_layers*np.max(Network.delay_max_matrix)) # Number of samples that will be recorded

if generate_data_mode == 'F':
    running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
else:
    running_period = (300*no_samples_per_cascade/10.0)  # Total running time in mili seconds

sim_window = round(1+running_period*10)                     # This is the number of iterations performed within each cascade

theta = 0.005                                               # The update threshold of the neurons in the network
sparse_thr0 = 0.0001
#------------------------------------------------------------------------------

#-------------------------Initialize Inference Parameters----------------------

#............................Perceptron-Based Approach.........................
if (inference_method == 3) or (inference_method == 2):    
    bin_size = 5
    d_window = 15    
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
    if (generate_data_mode == 'F'):
        Neural_Spikes,T_max = read_spikes(file_name_base,Network.no_layers,Network.n_exc_array,Network.n_inh_array,['c',no_stimul_rounds,sim_window])
    else:
        Neural_Spikes,T_max = read_spikes(file_name_base,Network.no_layers,Network.n_exc_array,Network.n_inh_array,['u',int(running_period*10)])
    #--------------------------------------------------------------------------
    
    if (generate_data_mode == 'F'):
        T_max = int(T_max)
    else:
        T_max = int(1000*T_max)
        
    T_step = int(T_max/6.0)
    T_range = range(T_step, T_max+1, T_step)
    
    
#==============================================================================

    

#============================INFER THE CONNECTIONS=============================
       
    
    if we_know_location.lower() == 'n':        
        W_tot,DD_tot = combine_weight_matrix(Network)    
        

    itr_T = 0
    #--------------------------------------------------------------------------           
    
    for T in T_range:
        
        #-------------------------If We Know the Location----------------------
        if we_know_location.lower() == 'y':
            
            for l_out in range(1,Network.no_layers):        
                    
                #~~~~~~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~~~~~~
                in_spikes,out_spikes = combine_spikes_matrix_FF(Network,l_out,generate_data_mode,T,Neural_Spikes)
                
                n,TT = in_spikes.shape
                m,TT = out_spikes.shape
                
                W_estimated = np.zeros([n,m])
                fixed_entries = np.zeros([n,m])
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                #~~~~~~~~~~~~~~~~~~~~Perfrom Inference~~~~~~~~~~~~~~~~~~~~~
                for infer_itr in range(0,infer_itr_max):
                    W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes,out_spikes,inference_method,inferece_params,W_estimated,1,generate_data_mode)                        
                        
                    W_temp = np.ma.masked_array(W_inferred_our_tot,mask= fixed_entries)                        
                    max_val = abs(W_temp).max()
                    min_val = W_temp.min()
                    #W_temp = 0.001 + (W_temp.astype(float) - max_val) * 0.006 / float(max_val - min_val)
                    W_estimated= []
                    centroids = []
                    #W_estimated,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,1.25*(1+infer_itr/7.5),2.5*(1+infer_itr/15.0),],0)
                        
                    if infer_itr < infer_itr_max-1:
                        fixed_entries = 1-isnan(W_estimated).astype(int)
                        
                    if norm(1-fixed_entries) == 0:
                        break

                    
                #W_temp = W_temp/float(max(abs(max_val),abs(min_val)))/1000.0
                W_temp = W_temp/float(max_val)/1000.0
                W_inferred_our_tot = W_temp.data
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                 
                #..............................................................                
                n_so_far = 0
                for l_in in range(0,l_out):
                    n_exc = Network.n_exc_array[l_in]
                    n_inh = Network.n_inh_array[l_in]
                    n = n_exc + n_inh                        
                    
                #...................Save the Belief Matrices...................                    
                    W_inferred_our = W_inferred_our_tot[n_so_far:n_so_far+n,:]                        
                    n_so_far = n_so_far + n
                    file_name_ending_base = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)                
                    file_name_ending = generate_file_name(file_name_ending_base,inference_method,we_know_location,pre_synaptic_method,generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,delay_known_flag)                    
                    file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
                    np.savetxt(file_name,W_inferred_our,'%1.5f',delimiter='\t')
                #..............................................................

        #----------------------------------------------------------------------
        
        else:
            
            #.....................Assign Inference Parameters......................
            out_spikes_tot,out_spikes_tot_mat,non_stimul_inds = combine_spikes_matrix(Network,T,generate_data_mode,Neural_Spikes)
            
            n,m = W_tot.shape
            W_estimated = np.zeros([n,m])
            fixed_entries = np.zeros([n,m])
            #......................................................................
        
            #........................Perfrom Inference.........................
            for infer_itr in range(0,infer_itr_max):
                W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(out_spikes_tot_mat,out_spikes_tot_mat,inference_method,inferece_params,W_estimated,0,generate_data_mode)
                
            #...................Save the Belief Matrices...................
            file_name_ending = generate_file_name(Network.file_name_ending,inference_method,we_know_location,pre_synaptic_method,generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,delay_known_flag)                    
            file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
            np.savetxt(file_name,W_inferred_our_tot,'%1.5f',delimiter='\t')
            #..............................................................

        itr_T = itr_T + 1        
        #----------------------------------------------------------------------
