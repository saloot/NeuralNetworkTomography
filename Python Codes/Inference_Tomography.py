#=======================DEFAULT VALUES FOR THE VARIABLES=======================
FRAC_STIMULATED_NEURONS_DEFAULT = 0.4
NO_STIMUL_ROUNDS_DEFAULT = 2000
ENSEMBLE_SIZE_DEFAULT = 1
FILE_NAME_BASE_DATA_DEFAULT = "./Data"
FILE_NAME_BASE_RESULT_DEFAULT = "./Results"
ENSEMBLE_COUNT_INIT_DEFAULT = 0
INFERENCE_METHOD_DEFAULT = 0
BINARY_MODE_DEFAULT = 4
SPARSITY_FLAG_DEFAULT = 0
GENERATE_DATA_MODE_DEFAULT = 'F'
INFERENCE_ITR_MAX_DEFAULT = 1
WE_KNOW_LOCATION_DEFAULT = 'N'
PRE_SYNAPTIC_NEURON_DEFAULT = 'A'
DELAY_KNOWN_DEFAULT = 'N'
VERIFY_FLAG_DEFAULT = 0
#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import os
import sys,getopt,os
import matplotlib.pyplot as plt
import pdb
from copy import deepcopy

import auxiliary_functions
reload(auxiliary_functions)
from auxiliary_functions import *

import Neurons_and_Networks
reload(Neurons_and_Networks)
from Neurons_and_Networks import NeuralNet
from Neurons_and_Networks import *
#==============================================================================

#================================INITIALIZATIONS===============================
n_exc_array = None; n_inh_array= None; connection_prob_matrix = None
random_delay_flag = None; no_layers = None; delay_max_matrix = None

plot_flag = 0

os.system('clear')                                              # Clear the commandline window
t0 = time()                                                # Initialize the timer
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:G:X:Y:K:C:V:J:")
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
        elif opt == '-R':
            random_delay_flag = int(arg)                        # The ensemble to start simulations from            
        elif opt == '-B':
            binary_mode = int(arg)                              # Defines the method to transform the graph to binary. "1" for threshold base and "2" for sparsity based                        
        elif opt == '-M':
            inference_method = int(arg)                         # The inference method
        elif opt == '-G':
            generate_data_mode = str(arg)                       # The data generating method            
        elif opt == '-Y':
            sparsity_flag = int(arg)                            # The flag that determines if sparsity should be observed during inference
        elif opt == '-X':
            infer_itr_max = int(arg)                            # The flag that determines if sparsity should be observed during inference            
        elif opt == '-K':
            we_know_location = str(arg)                         # The flag that determines if we know the location of neurons (with respect to each other) (Y/N)
        elif opt == '-C': 
            pre_synaptic_method = str(arg)                      # The flag that determines if all previous-layers neurons count as  pre-synaptic (A/O)
        elif opt == '-V': 
            verify_flag = int(arg)                              # If 1, the post-synaptic states will be predicted
        elif opt == '-J': 
            delay_known_flag = str(arg)                              # If 'Y', we assume that the delay is known during the inference algorithm
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
    file_name_base_data = FILE_NAME_BASE_DATA_DEFAULT;
    print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_data))

if 'ensemble_count_init' not in locals():
    ensemble_count_init = ENSEMBLE_COUNT_INIT_DEFAULT;
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))
    
if 'binary_mode' not in locals():
    binary_mode = BINARY_MODE_DEFAULT;
    print('ATTENTION: The default value of %s for binary_mode is considered.\n' %str(binary_mode))

if 'file_name_base_results' not in locals():
    file_name_base_results = FILE_NAME_BASE_RESULT_DEFAULT;
    print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_results))

if 'inference_method' not in locals():
    inference_method = INFERENCE_METHOD_DEFAULT;
    print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(inference_method))

if 'sparsity_flag' not in locals():
    sparsity_flag = SPARSITY_FLAG_DEFAULT;
    print('ATTENTION: The default value of %s for sparsity_flag is considered.\n' %str(sparsity_flag))

if 'generate_data_mode' not in locals():
    generate_data_mode = GENERATE_DATA_MODE_DEFAULT
    print('ATTENTION: The default value of %s for generate_data_mode is considered.\n' %str(generate_data_mode))

if 'infer_itr_max' not in locals():
    infer_itr_max = INFERENCE_ITR_MAX_DEFAULT
    print('ATTENTION: The default value of %s for infer_itr_max is considered.\n' %str(infer_itr_max))
    
if 'we_know_location' not in locals():
    we_know_location = WE_KNOW_LOCATION_DEFAULT
    print('ATTENTION: The default value of %s for we_know_location is considered.\n' %str(we_know_location))

if 'pre_synaptic_method' not in locals():
    pre_synaptic_method = PRE_SYNAPTIC_NEURON_DEFAULT
    print('ATTENTION: The default value of %s for pre_synaptic_method is considered.\n' %str(pre_synaptic_method))

if 'verify_flag' not in locals():
    verify_flag = VERIFY_FLAG_DEFAULT
    print('ATTENTION: The default value of %s for verify_flag is considered.\n' %str(verify_flag))
    
if 'delay_known_flag' not in locals():
    delay_known_flag = DELAY_KNOWN_DEFAULT
    print('ATTENTION: The default value of %s for delay_known_flag is considered.\n' %str(delay_known_flag))
    
#------------------------------------------------------------------------------

#--------------------------Initialize the Network------------------------------
#Network = NeuralNet(no_layers,n_exc_array,n_inh_array,connection_prob_matrix,delay_max_matrix,random_delay_flag,'')
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


if (generate_data_mode == 'F'):
    T_step = int(no_stimul_rounds/10.0)
    T_range = range(200, no_stimul_rounds, T_step)                 # The range of sample sizes considered to investigate the effect of sample size on the performance
else:
    T_step = int((running_period*1-100)/5.0)-1
    T_range = range(100, int(running_period*1)+1, T_step)
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if Necessary---------------
if not os.path.isdir(file_name_base_results):
    os.makedirs(file_name_base_results)
if not os.path.isdir(file_name_base_results+'/Accuracies'):
    temp = file_name_base_results + '/Accuracies'
    os.makedirs(temp)
if not os.path.isdir(file_name_base_results+'/RunningTimes'):
    temp = file_name_base_results + '/RunningTimes'
    os.makedirs(temp)
if not os.path.isdir(file_name_base_results+'/Inferred_Graphs'):
    temp = file_name_base_results + '/Inferred_Graphs'
    os.makedirs(temp)
if not os.path.isdir(file_name_base_results+'/Verified_Spikes'):
    temp = file_name_base_results + '/Verified_Spikes'
    os.makedirs(temp)

if not os.path.isdir(file_name_base_results+'/BeliefQuality'):    
    temp = file_name_base_results + '/BeliefQuality'
    os.makedirs(temp)
if not os.path.isdir(file_name_base_results+'/Plot_Results'):    
    temp = file_name_base_results + '/Plot_Results'
    os.makedirs(temp)    
#------------------------------------------------------------------------------

print T_range
t_base = time()
t_base = t_base-t0
alpha0 = 0.0000095
sparse_thr0 = 0.0001
adj_fact_exc = 0.75
adj_fact_inh = 0.5
#-------------------------Initialize Inference Parameters----------------------

#............................Correlation-Bases Approach........................
if (inference_method == 0):
    d_window = 0.015
    inferece_params = [1,theta,d_window,[]]
#..............................................................................

#............................Perceptron-Based Approach.........................
elif (inference_method == 3) or (inference_method == 2):
    #~~~~~We Know the Location of Neurons + Stimulate and Fire~~~~~
    # Current version
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
    #~~~We Know the Location of Neurons + Continuous Stimulation~~~
    # Just give the integrated version to the inference algorithm, i.e.: sum(in_spikes[stimul_span],out_spike)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if delay_known_flag == 'Y':
        d_window = 12
    else:
        d_window = 15
    #inferece_params = [alpha0,sparse_thr0,sparsity_flag,theta,120,generate_data_mode,[],d_window,[]]
    inferece_params = [alpha0,sparse_thr0,sparsity_flag,theta,250,generate_data_mode,[],d_window,[],[]]
#..............................................................................

#.............................Event-Based Approach.............................
elif (inference_method == 5):
    W_binary = []
    inferece_params = [alpha0,sparse_thr0,sparsity_flag,theta,sim_window]
#..............................................................................
            
#...............................CCF-Based Approach.............................
elif (inference_method == 4):
    d_window = 15
    inferece_params = [d_window]            
#..............................................................................

#...............Perceptron-Based Approach for Background Traffic...............
elif (inference_method == 6):
    d_window = 0.015
    inferece_params = [alpha0,sparse_thr0,sparsity_flag,theta,150,d_window,[],[]]
#..............................................................................

#............................Hebbian-Based Approach............................
else:
    inferece_params = [1]
    #W_inferred_our_tot = inference_alg_per_layer((-pow(-1,np.sign(cumulative_spikes_temp))),recorded_spikes_temp,inference_method,inferece_params,generate_data_mode,delay_known_flag)
#..............................................................................

#------------------------------------------------------------------------------

n_layer_1 = Network.n_exc_array[0]+Network.n_inh_array[0]
n_layer_2 = Network.n_exc_array[1]+Network.n_inh_array[1]

mean_void_b = np.zeros([len(T_range),n_layer_1])
mean_void_p = np.zeros([len(T_range),n_layer_1])
mean_void_r = np.zeros([len(T_range),n_layer_2])
mean_exc = np.zeros([len(T_range),n_layer_2])
mean_inh = np.zeros([len(T_range),n_layer_2])
mean_void = np.zeros([len(T_range),n_layer_2])

std_void_b = np.zeros([len(T_range),n_layer_1])
std_void_p = np.zeros([len(T_range),n_layer_1])
std_void_r = np.zeros([len(T_range),n_layer_2])
std_exc = np.zeros([len(T_range),n_layer_2])
std_inh = np.zeros([len(T_range),n_layer_2])
std_void = np.zeros([len(T_range),n_layer_2])
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
    
    T_max = int(1000*T_max)
    T_step = int(T_max/6.0)
    T_range = range(T_step, T_max+1, T_step)
    
#==============================================================================

    

#============================INFER THE CONNECTIONS=============================
       
    #--------------------------In-Loop Initializations-------------------------
    if (generate_data_mode == 'R'):
        #------------------Preprocess the Spikes if Necessary-----------------
        d_max = 10
        d_window = 1+d_max
        Processed_Neural_Spikes = {}
        Rough_Neural_Spikes = {}
        Rough_Neural_Times = {}
        Actual_Neural_Times = {}
        for l in range(0,Network.no_layers):                
            temp_list = Neural_Spikes[str(l)]
            spikes_actual_times = Neural_Spikes['act_'+str(l)]
            spikes = temp_list[2]
            n,T = spikes.shape
            out_spikes = np.zeros([n,T])
            out_spikes_rough = np.zeros([n,1+int(T/float(d_window))])
            spikes_rough_times = np.zeros([n,1+int(T/float(d_window))])
            t_rough = 0
            for t in range(0,T):
                out_spikes[:,t] = sum(spikes[:,max(t-d_window,0):t],axis = 1)
                if (np.mod(t,d_window) == 0) and (t > 0):
                    out_spikes_rough[:,t_rough] = out_spikes[:,t]
                    ind = np.nonzero(spikes[:,max(t-d_window,0):t])
                    if ind or (ind == 0):
                        spikes_rough_times[ind[0],t_rough] = ind[1]
                    t_rough = t_rough + 1
            Processed_Neural_Spikes[str(l)] = (out_spikes>0).astype(int)
            Rough_Neural_Spikes[str(l)] = (out_spikes_rough>0).astype(int)
            Rough_Neural_Times[str(l)] = spikes_rough_times
            Actual_Neural_Times[str(l)] = spikes_actual_times
    #--------------------------------------------------------------------------
    
    
    
    first_flag2 = 1
    itr_T = 0
    #--------------------------------------------------------------------------           
    
    for T in T_range:
        
        
        if inference_method == 0:
            inferece_params = [1/float(T),theta,d_window,generate_data_mode]           
        #-------------------------If We Know the Location----------------------
        if we_know_location.lower() == 'y':
            
            for l_out in range(1,Network.no_layers):        
                n_exc = Network.n_exc_array[l_out]
                n_inh = Network.n_inh_array[l_out]
                m = n_exc + n_inh                
                temp_list = Neural_Spikes[str(l_out)]            
                out_spikes_orig = (temp_list[2])
                out_spikes = out_spikes_orig
                if (generate_data_mode != 'R') and (inference_method != 4):
                    out_spikes = (out_spikes>0).astype(int)
                out_spikes = out_spikes[:,0:T]
                if (generate_data_mode == 'R'):
                    rough_out_spikes = Rough_Neural_Spikes[str(l_out)]
                    tt = 1+int(T/float(d_window))
                    rough_out_spikes = rough_out_spikes[:,0:tt]
                #.....If ALL previous Layers Count as Pre-synaptic Neurons.....        
                if pre_synaptic_method.lower() == 'a':
                    in_spikes = []
                    rough_in_spikes = []
                    

                    #~~~~~~~~~~~~~~Concatenate Pre-synaptic Spikes~~~~~~~~~~~~~
                    n_tot = 0
                    for l_in in range(0,l_out):
                        if (generate_data_mode != 'R'):
                            temp_list = Neural_Spikes[str(l_in)]
                            temp_list = temp_list[2]                            
                            
                        else:
                            temp_list = Processed_Neural_Spikes[str(l_in)]
                            temp_list = Actual_Neural_Times[str(l_in)]
                        n_exc = Network.n_exc_array[l_in]
                        n_inh = Network.n_inh_array[l_in]
                        n = n_exc + n_inh
                        n_tot = n_tot + n_exc + n_inh
                        if len(in_spikes):
                            in_spikes_temp = temp_list
                            in_spikes_temp = in_spikes_temp[:,0:T]
                            in_spikes = np.concatenate([in_spikes,in_spikes_temp])
                            if (generate_data_mode == 'R'):
                                temp_sp = Rough_Neural_Spikes[str(l_in)]                                
                                temp_sp = temp_sp[:,0:tt]
                                rough_in_spikes = np.concatenate([rough_in_spikes,temp_sp])
                        else:
                            if l_in == 0:
                                in_spikes = temp_list
                        
                        
                            
                                
                            if (generate_data_mode != 'R'):
                                in_spikes = in_spikes[:,0:T]
                            if (generate_data_mode == 'R'):
                                temp_sp = Rough_Neural_Spikes[str(l_in)]                                
                                rough_in_spikes = temp_sp[:,0:tt]
                                
                    if (generate_data_mode != 'R'):
                        in_spikes_orig = in_spikes
                    else:
                        in_spikes_orig = rough_in_spikes
                        out_spikes_tot = deepcopy(Actual_Neural_Times[str(l_out)])
                        in_spikes_tot = deepcopy(Actual_Neural_Times[str(l_in)])
                        in_spikes_tot_mat = np.zeros([n,T])
                        out_spikes_tot_mat = np.zeros([m,T])
                        for sps in in_spikes_tot:
                            sps_times = np.array(in_spikes_tot[sps])
                            sps_times = sps_times[sps_times<T/1000.0]
                            iik = int(sps)
                            
                            sps_inds = (1000*sps_times).astype(int)
                            in_spikes_tot_mat[iik,sps_inds] = 1
                            #aas = sps_times<=T/1000.0
                            #aas = np.nonzero(aas)[0]
                            #if sum(aas):
                            #    aas = aas[len(aas)-1]
                            #    sps_times = sps_times[0:aas]
                            #else:
                            #    sps_times = []
                            in_spikes_tot[sps] = sps_times
                        
                        for sps in out_spikes_tot:
                            sps_times = np.array(out_spikes_tot[sps])
                            sps_times = sps_times[sps_times<T/1000.0]
                            iik = int(sps)
                            
                            sps_inds = (1000*sps_times).astype(int)
                            out_spikes_tot_mat[iik,sps_inds] = 1
                            #aas = sps_times<=T/1000.0
                            #aas = np.nonzero(aas)[0]
                            #if sum(aas):
                            #    aas = aas[len(aas)-1]
                            #    sps_times = sps_times[0:aas]
                            #else:
                            #    sps_times = []
                            out_spikes_tot[sps] = sps_times
                            
                        
                        out_spikes = out_spikes_tot
                        in_spikes = in_spikes_tot
                        
                        #pdb.set_trace()
                        
                    if (generate_data_mode != 'R'):# or (inference_method != 4):
                        in_spikes = (in_spikes>0).astype(int)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~~~~~~
                    n = n_tot   
                    W_estimated = np.zeros([n,m])
                    fixed_entries = np.zeros([n,m])
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~~~~Perfrom Inference~~~~~~~~~~~~~~~~~~~~~
                    ind = str(l_in) + str(l_out);temp_list = Network.Neural_Connections[ind];W = temp_list[0];DD = temp_list[1]                    
                    if inference_method == 6 or inference_method == 3:
                        if delay_known_flag == 'Y':
                            fixed_entries = (DD >0).astype(int)
                            d_max = np.max(Network.delay_max_matrix)
                            Dels = np.multiply(fixed_entries,1000*DD) + np.multiply(1-fixed_entries,np.random.uniform(0,d_max,[n,m]))
                            inferece_params[6] = Dels
                        else:                            
                            Dels = float('nan')*DD
                    
                
                        if inference_method == 3:
                            inferece_params[8] = Dels
                        else:
                            inferece_params[6] = Dels
                            
                    if (inference_method == 4) and (generate_data_mode == 'R'):
                        out_spikes = out_spikes_tot_mat
                        in_spikes = in_spikes_tot_mat
                    
                    if (inference_method == 3) and (generate_data_mode == 'R'):
                        out_spikes = out_spikes_tot_mat
                        in_spikes = in_spikes_tot_mat                        
                        
                    #pdb.set_trace()
                    for infer_itr in range(0,infer_itr_max):
                        W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes,out_spikes,inference_method,inferece_params,W_estimated,1,generate_data_mode,delay_known_flag)                        
                        # W_bin = 0.001*(W_inferred_our_tot>W_inferred_our_tot.mean()+W_inferred_our_tot.std()).astype(int) - 0.005*(W_inferred_our_tot<-W_inferred_our_tot.mean()-W_inferred_our_tot.std()).astype(int)
                        #recal,precision = calucate_accuracy(W_estimated,W)
                        #print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
                        #print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
                        #print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
                        #print '\n'
                        
                        W_temp = np.ma.masked_array(W_inferred_our_tot,mask= fixed_entries)                        
                        max_val = abs(W_temp).max()
                        min_val = W_temp.min()
                        #W_temp = 0.001 + (W_temp.astype(float) - max_val) * 0.006 / float(max_val - min_val)
                        
                        #pdb.set_trace()                        
                        W_estimated= []
                        centroids = []
                        #W_estimated,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,1.25*(1+infer_itr/7.5),2.5*(1+infer_itr/15.0),],0)
                        
                        
                        if infer_itr < infer_itr_max-1:
                            fixed_entries = 1-isnan(W_estimated).astype(int)
                        
                        if norm(1-fixed_entries) == 0:
                            break
                        #if norm(cost) and (cost[len(cost)-1] == 0):
                        #    break
                    
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
                
                #..If Only One previous Layer Counts as Pre-synaptic Neurons...                    
                    if pre_synaptic_method.lower() == 'o':
                        n_so_far = 0
                        temp_list = Neural_Spikes[str(l_in)]            
                        in_spikes = (temp_list[2])
                        in_spikes = in_spikes[:,0:T]
                        in_spikes_orig = in_spikes
                        if (generate_data_mode != 'R') and (inference_method != 4):
                            in_spikes = (in_spikes>0).astype(int)

                        #~~~~~~~~~~~~~~~~~~Perfrom Inference~~~~~~~~~~~~~~~~~~~
                        for infer_itr in range(0,infer_itr_max):
                            W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes,out_spikes,inference_method,inferece_params,W_estimated,1,generate_data_mode,delay_known_flag)
                            W_inferred_our_tot = W_inferred_our_tot/(abs(W_inferred_our_tot).max())
                            W_estimated= []
                            centroids = []
                            #W_estimated,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_ind],0)#
                            fixed_ind = 1-isnan(W_estimated).astype(int)
                            fixed_ind = fixed_ind.reshape(n,m)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                    
                #..............................................................   
                    
                #...................Save the Belief Matrices...................
                    ind = str(l_in) + str(l_out)
                    temp_list = Network.Neural_Connections[ind]
                    W = temp_list[0]
                    
                    W_inferred_our = W_inferred_our_tot[n_so_far:n_so_far+n,:]                        
                    n_so_far = n_so_far + n
                
                    file_name_ending23 = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                    file_name_ending23 = file_name_ending23 + '_I_' + str(inference_method)
                    file_name_ending23 = file_name_ending23 + '_Loc_' + we_know_location
                    file_name_ending23 = file_name_ending23 + '_Pre_' + pre_synaptic_method
                    file_name_ending23 = file_name_ending23 + '_G_' + generate_data_mode
                    file_name_ending23 = file_name_ending23 + '_X_' + str(infer_itr_max)
                    file_name_ending23 = file_name_ending23 + '_Q_' + str(frac_stimulated_neurons)
                    if (sparsity_flag):
                        file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
                    file_name_ending2 = file_name_ending23 +"_T_%s" %str(T)
                    if delay_known_flag == 'Y':
                        file_name_ending2 = file_name_ending2 +"_DD_Y"

                    file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending2
                    np.savetxt(file_name,W_inferred_our,'%1.5f',delimiter='\t')
                #..............................................................
                
                
                #.....................Transform to Binary......................
                    #fixed_entries = 1- np.sign(Updated_Vals)                    
                    params = [adj_fact_exc,adj_fact_inh,fixed_entries]
                    
                    
                    if binary_mode == 7:
                        W_bin,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,0.5*(1+infer_itr/4.0),1.0*(1+infer_itr/2.0)],0)                    
                        for i in range(0,m):
                            for j in range(0,n):
                                if isnan(W_bin[j,i]):
                                    W_bin[j,i] = 0
                    else:
                        W_inferred_our_tot = W_inferred_our_tot
                        W_bin = []
                        centroids = []
                        
                        #W_bin,centroids = beliefs_to_binary(binary_mode,1000*W_inferred_our_tot,params,0)
                        #centroids = np.vstack([centroids,np.zeros([3])])
                        
                    #pdb.set_trace()
                    file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
                    file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                    file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
                
                    file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2
                    np.savetxt(file_name,W_bin,'%1.5f',delimiter='\t',newline='\n')
        
                
                    file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_%s.txt" %file_name_ending2                
                    ww = W_inferred_our.ravel()
                    ww = np.vstack([ww,np.zeros([len(ww)])])
                    np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')

                    if (binary_mode == 4):
                        file_name = file_name_base_results + "/Inferred_Graphs/Centroids_%s.txt" %file_name_ending2
                        np.savetxt(file_name,centroids,'%f',delimiter='\t')
                #..............................................................
                
                #....................Calculate the Accuracy....................
                    #for i in range(0,m):
                    #    for j in range(0,n):
                    #        if isnan(W_estimated[j,i]):
                    #            W_estimated[j,i] = 0
                    #recal,precision = calucate_accuracy(W_bin,W) #(W_estimated,W)
                    recal = [0,0,0]
                    precision = [0,0,0]
                    print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
                    #print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
                    #print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
                    #print '\n'
                #..............................................................
                    
                    
                #.............Calculate Spike Prediction Accuracy..............
                    if verify_flag: #(T == T_range[len(T_range)-1]):
                        Network_in = {}
                        Network_in['n_in'] = n
                        Network_in['n_out'] = m
                        Network_in['d_max'] = 1
                        if (generate_data_mode == 'R'):
                            TT = 1+int(T/float(d_window))
                            out_spikes_orig = rough_out_spikes
                        else:
                            TT = T
                            out_spikes_orig = out_spikes
                            
                        sps_pred = np.zeros([m,TT])
                        W_bin = np.multiply(W_bin>0,0.001*np.ones([n,m])) + np.multiply(W_bin<0,-0.005*np.ones([n,m]))                        
                        
                        temp_W = W_bin
                        temp_W = temp_W.reshape([n,m])
                        Network_in['W'] = temp_W
                        Network_in['D'] = np.sign(abs(temp_W))/10000.0                    
                        
                        
                        sps_pred = verify_neural_activity_simplified(temp_W,in_spikes_orig,theta)
                        #for t in range(0,TT):
                        #    sps_in = in_spikes_orig[:,t]
                        #    stimulated_neurons = np.nonzero(sps_in)
                        #    stimulated_neurons = stimulated_neurons[0]                            
                        #    stimulation_times = sps_in[stimulated_neurons] * 1000.0
                        #    Neural_Connections_Out,out_spike = verify_neural_activity(Network,Network_in,(no_samples_per_cascade/10.0),frac_stimulated_neurons,stimulated_neurons,stimulation_times)
                        #    sps_pred[:,t] = out_spike                                
                            
                        
                        print sum(abs(np.sign(out_spikes_orig) - np.sign(sps_pred) ))
                                            
                        file_name = file_name_base_results + "/Verified_Spikes/Predicted_Spikes_%s.txt" %file_name_ending2
                        save_matrix = np.zeros([10*min(T,100),3])                        
                        itr_mat = 0
                        for ik in range(0,10):
                            for ij in range(0,min(TT,100)):
                                val = 1 + np.sign(out_spikes_orig[ik,ij]) - np.sign(sps_pred[ik,ij])
                                
                                save_matrix[itr_mat,:] = [ij,ik,val]
                                itr_mat = itr_mat + 1
                        
                        np.savetxt(file_name,save_matrix,'%3.5f',delimiter='\t')
                        
                        file_name = file_name_base_results + "/Verified_Spikes/Average_Mismatch_%s.txt" %file_name_ending2
                        avg_mismatch = sum(abs(np.sign(out_spikes_orig) - np.sign(sps_pred) ),axis = 1)
                        avg_mismatch = avg_mismatch.reshape([m,1])
                        aa = np.array(range(0,m))
                        aa = aa.reshape([m,1])
                        avg_mismatch = np.hstack([aa,avg_mismatch])
                        np.savetxt(file_name,avg_mismatch,'%3.5f',delimiter='\t')
                #..............................................................    
                    
                #......................Save the Accuracies......................
                    temp_ending = file_name_ending2.replace("_T_%s" %str(T),'')
                    file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %temp_ending
                    if (T == T_range[0]):
                        acc_file = open(file_name,'w')
                    else:
                        acc_file = open(file_name,'a')
                    acc_file.write("%d \t %f \t %f \t %f" %(T,recal[0],recal[1],recal[2]))
                    acc_file.write("\n")
                    acc_file.close()
                        
                    file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %temp_ending
                    if (T == T_range[0]):
                        acc_file = open(file_name,'w')
                    else:
                        acc_file = open(file_name,'a')
                    acc_file.write("%d \t %f \t %f \t %f" %(T,recal[0],recal[1],recal[2]))
                    acc_file.write("\n")
                    acc_file.close()
                    
                    if verify_flag:
                        file_name = file_name_base_results + "/Verified_Spikes/Spike_Acc_per_T_%s.txt" %temp_ending
                        if (T == T_range[0]):
                            acc_file = open(file_name,'w')
                        else:
                            acc_file = open(file_name,'a')
                        acc_file.write("%d \t %f \n" %(T,sum(abs(np.sign(out_spikes_orig) - np.sign(sps_pred) ))/float(sum(np.sign(out_spikes_orig))) ))
                        acc_file.write("\n")
                        acc_file.close()
                    
                    
                    if (T == T_range[len(T_range)-1]):                        
                        file_name_ending2 = file_name_ending2.replace('_l_' + str(l_in) + '_to_' + str(l_out),'')        
                        file_name = file_name_base_results + "/Accuracies/Rec_Layers_Segregated_%s.txt" %file_name_ending2
                        if (first_flag2):
                            acc_file = open(file_name,'w')
                        else:
                            acc_file = open(file_name,'a')
                
                        acc_file.write("%s \t %f \t %f \n" %(ind,recal[0],recal[1]))
                        acc_file.close()
        
                        file_name = file_name_base_results + "/Accuracies/Prec_Layers_Segregated_%s.txt" %file_name_ending2
                        if (first_flag2):
                            acc_file = open(file_name,'w')
                        else:
                            acc_file = open(file_name,'a')                        
                
                        acc_file.write("%s \t %f \t %f \n" %(ind,precision[0],precision[1]))
                        acc_file.close()
                        
                        if ( (l_in == Network.no_layers-1) and (l_out == Network.no_layers-1) ):
                            first_flag2 = 0
        #----------------------------------------------------------------------
        
        else:
            
            #............Construct the Concatenated Weight Matrix..............            
            W_tot = []
            DD_tot = []
            if generate_data_mode == 'R':
                rough_spikes_tot = {}
                in_spikes_tot = {}
                out_spikes_tot = {}
            else:
                in_spikes_tot = []
                out_spikes_tot = []
            
            n_so_far = 0
            n_tot = 0
            for l_in in range(0,Network.no_layers):
                n_exc = Network.n_exc_array[l_in]
                n_inh = Network.n_inh_array[l_in]
                n = n_exc + n_inh
                n_tot = n_tot + n
            out_spikes_tot_mat = np.zeros([n_tot,T])
            for l_in in range(0,Network.no_layers):
                n_exc = Network.n_exc_array[l_in]
                n_inh = Network.n_inh_array[l_in]
                n = n_exc + n_inh
                
                
                if generate_data_mode == 'R':
                    
                    spikes_times = Actual_Neural_Times[str(l_in)]
                    for i in range(0,n):
                        ind = str(n_so_far)
                        ttimes = np.array(spikes_times[str(i)])
                        ttimes = ttimes[ttimes<T/1000.0]
                        #if sum(abs(ttimes)):
                        #    ttimes = ttimes[ttimes<T/1000.0]
                        out_spikes_tot[ind] = ttimes
                                                
                            
                        sps_inds = (1000*ttimes).astype(int)
                        out_spikes_tot_mat[n_so_far,sps_inds] = 1
                        
                        n_so_far = n_so_far + 1
                                        
                else:
                    temp_list = Neural_Spikes[str(l_in)]
                    spikes = (temp_list[2])
                    #rough_spikes = Rough_Neural_Times[str(l_in)]
                    
                    #spikes = (spikes>0).astype(int)
                    if len(out_spikes_tot):
                        out_spikes_tot = np.vstack([out_spikes_tot,spikes])
                        if generate_data_mode == 'R':
                            rough_spikes_tot = np.vstack([rough_spikes_tot,rough_spikes])
                    else:
                        out_spikes_tot = spikes
                        if generate_data_mode == 'R':
                            rough_spikes_tot = rough_spikes
                
                    
                W_temp = []
                D_temp = []
                for l_out in range(0,Network.no_layers):
                    n_exc = Network.n_exc_array[l_out]
                    n_inh = Network.n_inh_array[l_out]
                    m = n_exc + n_inh
                        
                    if (l_out < l_in):
                        if len(W_temp):
                            W_temp = np.hstack([W_temp,np.zeros([n,m])])
                        else:
                            W_temp = np.zeros([n,m])
                            
                        if len(D_temp):
                            D_temp = np.hstack([D_temp,np.zeros([n,m])])
                        else:
                            D_temp = np.zeros([n,m])
                            
                    else:                                        
                        ind = str(l_in) + str(l_out);
                        temp_list = Network.Neural_Connections[ind];
                        W = temp_list[0]
                        DD = temp_list[1]
                        
                        if len(W_temp):
                            W_temp = np.hstack([W_temp,W])
                        else:
                            W_temp = W
                        
                        if len(D_temp):
                            D_temp = np.hstack([D_temp,DD])
                        else:
                            D_temp = DD
                
                
                
                if len(W_tot):
                    W_tot = np.vstack([W_tot,W_temp])
                else:
                    W_tot = W_temp
                
                if len(DD_tot):
                    DD_tot = np.vstack([DD_tot,D_temp])
                else:
                    DD_tot = D_temp
            #..................................................................
            
            #......................In-Loop Initializations.....................
            W = W_tot
            m,n = W.shape
            W_estimated = np.zeros([n,m])
            fixed_entries = np.zeros([n,m])
            if (generate_data_mode == 'R'):
                #tt = 1+int(T/float(d_window))
                #out_spikes_tot = rough_spikes_tot[:,0:tt]
                #in_spikes_tot = out_spikes_tot
                #in_spikes_orig = in_spikes_tot
                #out_spikes_tot = rough_spikes_tot
                in_spikes_tot = out_spikes_tot
            else:
                tt = T
                out_spikes_tot = out_spikes_tot[:,0:tt]
                in_spikes_tot = out_spikes_tot
                in_spikes_orig = in_spikes_tot            
            #..................................................................
            
            
            non_stimul_inds = {}
            if (generate_data_mode != 'R'):
                for ttt in range(0,T):
                    temp = in_spikes_tot[0:n_layer_1,ttt]
                    ttemp = np.nonzero(temp<=0)
                    ttemp = list(ttemp[0])
                    ttemp.extend(range(n_layer_1,n_layer_1+n_layer_2))
                    non_stimul_inds[str(ttt)] = np.array(ttemp)
                
            
            
                
            #pdb.set_trace()            
            if inference_method == 6 or inference_method == 3:
                if delay_known_flag == 'Y':
                    fixedentries = (DD_tot >0).astype(int)
                    d_max = np.max(Network.delay_max_matrix)
                    Dels = np.multiply(fixedentries,1000*DD_tot) + np.multiply(1-fixedentries,np.random.uniform(0,d_max,[n,m]))
                else:
                    Dels = float('nan')*DD
                    
                
                if inference_method == 3:
                    inferece_params[8] = Dels
                else:
                    inferece_params[6] = Dels
                    
                            
            if inference_method == 4:
                in_spikes_tot = out_spikes_tot_mat
                out_spikes_tot = out_spikes_tot_mat
                
            if (inference_method == 3) and (generate_data_mode == 'R'):
                in_spikes_tot = out_spikes_tot_mat
                out_spikes_tot = out_spikes_tot_mat
                inferece_params[len(inferece_params)-4] = non_stimul_inds                
                
            elif inference_method == 3:
                inferece_params[len(inferece_params)-4] = non_stimul_inds
            #........................Perfrom Inference.........................
            for infer_itr in range(0,infer_itr_max):
                
                #~~~~~~~~~~~~~~Classify Excitatory Neurons Only~~~~~~~~~~~~~~~~                
                W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes_tot,out_spikes_tot,inference_method,inferece_params,W_estimated,0,generate_data_mode,delay_known_flag)
                W_tmp = abs(np.ma.masked_array(W_inferred_our_tot,mask= (W_inferred_our_tot==float('inf')).astype(int)))
                #W_bin = 0.001*(W_inferred_our_tot>W_inferred_our_tot.mean()+W_inferred_our_tot.std()).astype(int) - 0.005*(W_inferred_our_tot<-W_inferred_our_tot.mean()-W_inferred_our_tot.std()).astype(int)
                
                
                #-----Calculate the Mean and Variance of Different Beliefs-----
                
                #.................FF Excitatory Connections....................                
                W_e = np.ma.masked_array(W_inferred_our_tot[0:n_layer_1,n_layer_1:n_layer_1+n_layer_2],mask= (W_tot[0:n_layer_1,n_layer_1:n_layer_1+n_layer_2]<=0).astype(int))
                mean_exc[itr_T,:] = mean_exc[itr_T,:] + W_e.mean(axis = 0).data
                std_exc[itr_T,:] = std_exc[itr_T,:] + W_e.std(axis = 0).data
                
                W_i = np.ma.masked_array(W_inferred_our_tot[0:n_layer_1,n_layer_1:n_layer_1+n_layer_2],mask= (W_tot[0:n_layer_1,n_layer_1:n_layer_1+n_layer_2]>=0).astype(int))
                mean_inh[itr_T,:] = mean_inh[itr_T,:] + W_i.mean(axis = 0).data
                std_inh[itr_T,:] = std_inh[itr_T,:] + W_i.std(axis = 0).data
                
                W_v = np.ma.masked_array(W_inferred_our_tot[0:n_layer_1,n_layer_1:n_layer_1+n_layer_2],mask= (W_tot[0:n_layer_1,n_layer_1:n_layer_1+n_layer_2]!=0).astype(int))
                mean_void[itr_T,:] = mean_void[itr_T,:] + W_v.mean(axis = 0).data
                std_void[itr_T,:] = std_void[itr_T,:] + W_v.std(axis = 0).data                
                #..............................................................
                                
                #.......Recurrent Connections in the Post-Syanptic Layer.......
                W_v_r = np.ma.masked_array(W_inferred_our_tot[n_layer_1:n_layer_1+n_layer_2,n_layer_1:n_layer_1+n_layer_2],mask= (W_tot[n_layer_1:n_layer_1+n_layer_2,n_layer_1:n_layer_1+n_layer_2]!=0).astype(int))
                mean_void_r[itr_T,:] =  mean_void_r[itr_T,:] + W_v_r.mean(axis = 0).data
                std_void_r[itr_T,:] = std_void_r[itr_T,:] + W_v_r.std(axis = 0).data                
                #..............................................................                
                
                #.......Recurrent Connections in the Pre-Syanptic Layer........
                W_v_p = np.ma.masked_array(W_inferred_our_tot[0:n_layer_1,0:n_layer_1],mask= (W_tot[0:n_layer_1,0:n_layer_1]!=0).astype(int))
                mean_void_p[itr_T,:] =  mean_void_p[itr_T,:] + W_v_p.mean(axis = 0).data
                std_void_p[itr_T,:] = std_void_p[itr_T,:] + W_v_p.std(axis = 0).data                
                #..............................................................
                
                #.....Backward Connections From Post to Pre-Syanptic Layer.....
                W_v_b = np.ma.masked_array(W_inferred_our_tot[n_layer_1:n_layer_1+n_layer_2,0:n_layer_1],mask= (W_tot[n_layer_1:n_layer_1+n_layer_2,0:n_layer_1]!=0).astype(int))
                mean_void_b[itr_T,:] =  mean_void_b[itr_T,:] + W_v_b.mean(axis = 0).data
                std_void_b[itr_T,:] = std_void_b[itr_T,:] + W_v_b.std(axis = 0).data                
                #..............................................................
                
                if 0:
                    plt.plot(mean_exc,'r'); plt.plot(mean_inh,'b');plt.plot(mean_void,'g');plt.plot(mean_void_r,'g--'); plt.show();
                    plt.plot(mean_void_b,'g');plt.plot(mean_void_p,'g--'); plt.show();
                    
                #--------------------------------------------------------------
                
                if (inference_method == 4):
                    for ii in range (0,m):
                        for jj in range(0,n):
                            if (W_inferred_our_tot[ii,jj]==float('inf')):
                                W_inferred_our_tot[ii,jj] = 1*W_tmp.max()
                                            
                
                #W_estimated,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,1.25*(1+infer_itr/7.5),2500*(1+infer_itr/15.0),],0)
                #fixed_entries = 1-isnan(W_estimated).astype(int)
                #pdb.set_trace()
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~~Classify Inhibitory Neurons Only~~~~~~~~~~~~~~~~                
                #W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes_tot,out_spikes_tot,inference_method,inferece_params,W_estimated,0,generate_data_mode,delay_known_flag)
                #W_tmp = abs(np.ma.masked_array(W_inferred_our_tot,mask= (W_inferred_our_tot==float('inf')).astype(int)))
                
                #if (inference_method == 4):
                #    for ii in range (0,m):
                #        for jj in range(0,n):
                #            if (W_inferred_our_tot[ii,jj]==float('inf')):
                #                W_inferred_our_tot[ii,jj] = 1*W_tmp.max()
                #            
                #W_temp = np.ma.masked_array(W_inferred_our_tot,mask= fixed_entries)
                
                #W_estimated_inh,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,12500*(1+infer_itr/7.5),2*(1+infer_itr/15.0),],0)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~~~~~~~~~~~~Mix the Two~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #for ii in range (0,m):
                #    for jj in range(0,n):
                #        if not np.isnan(W_estimated_inh[ii,jj]):
                #            W_estimated[ii,jj] = W_estimated_inh[ii,jj]
                        
                #        if (infer_itr == infer_itr_max -1):
                #            if np.isnan(W_estimated[ii,jj]):
                #                W_estimated[ii,jj] = 0
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                #if (infer_itr < infer_itr_max -1):
                #    fixed_entries = 1-isnan(W_estimated).astype(int)

                #if norm(1-fixed_entries) == 0:
                #    break
            #..................................................................
            
            
            #...................Save the Belief Matrices...................                                                
            file_name_ending23 = Network.file_name_ending + '_I_' + str(inference_method)
            file_name_ending23 = file_name_ending23 + '_Loc_' + we_know_location
            file_name_ending23 = file_name_ending23 + '_Pre_' + pre_synaptic_method
            file_name_ending23 = file_name_ending23 + '_G_' + generate_data_mode
            file_name_ending23 = file_name_ending23 + '_X_' + str(infer_itr_max)
            file_name_ending23 = file_name_ending23 + '_Q_' + str(frac_stimulated_neurons)
            if (sparsity_flag):
                file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
            file_name_ending2 = file_name_ending23 +"_T_%s" %str(T)
            if delay_known_flag == 'Y':
                file_name_ending2 = file_name_ending2 +"_DD_Y"

            file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending2
            np.savetxt(file_name,W_inferred_our_tot,'%1.5f',delimiter='\t')
            #..............................................................
                
                
            #.....................Transform to Binary......................            
            params = [adj_fact_exc,adj_fact_inh,fixed_entries]
            if binary_mode == 7:
                W_bin,centroids = beliefs_to_binary(7,W_estimated,[fixed_entries,0.5*(1+infer_itr/4.0),1.0*(1+infer_itr/2.0)],0)                    
                for i in range(0,m):
                    for j in range(0,n):
                        if isnan(W_bin[j,i]):
                            W_bin[j,i] = 0
            else:
                W_inferred_our_tot = W_inferred_our_tot
                W_bin = []
                centroids = []
                #W_bin,centroids = beliefs_to_binary(binary_mode,1000*W_inferred_our_tot,params,0)
            
            #W_bin = W_estimated
            
            file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
            file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
            file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
                
            file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2            
            np.savetxt(file_name,W_bin,'%1.5f',delimiter='\t',newline='\n')
            
                
            file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_%s.txt" %file_name_ending2                
            ww = W_inferred_our_tot.ravel()
            ww = np.vstack([ww,np.zeros([len(ww)])])
            np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')

            if 0:#(binary_mode == 4):
                file_name = file_name_base_results + "/Inferred_Graphs/Centroids_%s.txt" %file_name_ending2
                centroids = np.vstack([centroids,np.zeros([3])])
                np.savetxt(file_name,centroids,'%f',delimiter='\t')
            #..............................................................
                
            #....................Calculate the Accuracy....................
            #recal,precision = calucate_accuracy(W_bin,W)
            recal = [0,0,0]
            precision = [0,0,0]
            print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
            #print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
            #print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
            #print '\n'
            #..............................................................
            
            #pdb.set_trace()   
            #.............Calculate Spike Prediction Accuracy..............
            if verify_flag: #(T == T_range[len(T_range)-1]):
                Network_in = {}
                Network_in['n_in'] = n
                Network_in['n_out'] = m
                Network_in['d_max'] = 1
                
                W_bin = np.multiply(W_bin>0,0.001*np.ones([n,m])) + np.multiply(W_bin<0,-0.005*np.ones([n,m]))                        
                        
                temp_W = W_bin
                temp_W = temp_W.reshape([n,m])
                Network_in['W'] = temp_W
                Network_in['D'] = np.sign(abs(temp_W))/10000.0
                
                
                
                if (generate_data_mode == 'R'):                                        
                    out_spikes_tot = rough_spikes_tot[:,0:tt]
                    sps_pred = np.zeros([m,tt])
                    rough_out_spikes = rough_spikes_tot[:,0:tt]
                    for j in range(0,m):
                        sps_out = out_spikes_tot[j,:]
                        sps_out = sps_out.reshape([1,tt])
                        sps_temp = np.dot(np.ones([m,1]),sps_out)
                        sps_in = np.multiply((out_spikes_tot < sps_temp).astype(int),(out_spikes_tot>0).astype(int))
                        sps_pred[j,:] = verify_neural_activity_simplified(temp_W[:,j],sps_in,theta)
                    
                    print sum(abs(np.sign(out_spikes_tot) - np.sign(sps_pred) ))
                    
                else:
                    tt = T
                    sps_pred = np.zeros([m,T])
                    for t in range(0,T):                    
                        sps_in = np.multiply(in_spikes_orig[:,t],(in_spikes_orig[:,t] == 0.0002).astype(int))
                        stimulated_neurons = np.nonzero(sps_in)
                        stimulated_neurons = stimulated_neurons[0]
                        stimulation_times = sps_in[stimulated_neurons] * 1000.0
                        Neural_Connections_Out,out_spike = verify_neural_activity(Network,Network_in,running_period,frac_stimulated_neurons,stimulated_neurons,stimulation_times)
                        sps_pred[:,t] = out_spike + sps_in                    
                        
                    print sum(abs(np.sign(out_spikes_tot) - np.sign(sps_pred) ))
                                            
                file_name = file_name_base_results + "/Verified_Spikes/Predicted_Spikes_%s.txt" %file_name_ending2
                save_matrix = np.zeros([10*min(T,100),3])
                itr_mat = 0
                for ik in range(n-(n_exc+n_inh),n-(n_exc+n_inh)+10):
                    for ij in range(0,min(tt,100)):
                        val = 1 + np.sign(out_spikes_tot[ik,ij]) - np.sign(sps_pred[ik,ij])
                                
                        save_matrix[itr_mat,:] = [ij,ik,val]
                        itr_mat = itr_mat + 1
                        
                np.savetxt(file_name,save_matrix,'%3.5f',delimiter='\t')
                        
                file_name = file_name_base_results + "/Verified_Spikes/Average_Mismatch_%s.txt" %file_name_ending2
                avg_mismatch = sum(abs(np.sign(out_spikes_tot) - np.sign(sps_pred) ),axis = 1)
                avg_mismatch = avg_mismatch.reshape([m,1])
                aa = np.array(range(0,m))
                aa = aa.reshape([m,1])
                avg_mismatch = np.hstack([aa,avg_mismatch])
                np.savetxt(file_name,avg_mismatch,'%3.5f',delimiter='\t')
                #..............................................................
                
            #......................Save the Accuracies......................
            temp_ending = file_name_ending2.replace("_T_%s" %str(T),'')
            file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %temp_ending
            if (T == T_range[0]):
                acc_file = open(file_name,'w')
            else:
                acc_file = open(file_name,'a')
            acc_file.write("%d \t %f \t %f \t %f" %(T,recal[0],recal[1],recal[2]))
            acc_file.write("\n")
            acc_file.close()
                        
            file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %temp_ending
            if (T == T_range[0]):
                acc_file = open(file_name,'w')
            else:
                acc_file = open(file_name,'a')
            acc_file.write("%d \t %f \t %f \t %f" %(T,recal[0],recal[1],recal[2]))
            acc_file.write("\n")
            acc_file.close()
                        
            if verify_flag:
                file_name = file_name_base_results + "/Verified_Spikes/Spike_Acc_per_T_%s.txt" %temp_ending
                if (T == T_range[0]):
                    acc_file = open(file_name,'w')
                else:
                    acc_file = open(file_name,'a')
                acc_file.write("%d \t %f \n" %(tt,sum(abs(np.sign(out_spikes_tot[n-(n_exc+n_inh):n,:]) - np.sign(sps_pred[n-(n_exc+n_inh):n,:]) ))/float(sum(np.sign(out_spikes_tot[n-(n_exc+n_inh):n,:]))) ))                
                acc_file.close()
                    
            if (T == T_range[len(T_range)-1]):                        
                file_name_ending2 = file_name_ending2.replace('_l_' + str(l_in) + '_to_' + str(l_out),'')        
                file_name = file_name_base_results + "/Accuracies/Rec_Layers_Segregated_%s.txt" %file_name_ending2
                if (first_flag2):
                    acc_file = open(file_name,'w')
                else:
                    acc_file = open(file_name,'a')
                
                acc_file.write("%s \t %f \t %f \n" %(ind,recal[0],recal[1]))
                acc_file.close()
        
                file_name = file_name_base_results + "/Accuracies/Prec_Layers_Segregated_%s.txt" %file_name_ending2
                if (first_flag2):
                    acc_file = open(file_name,'w')
                else:
                    acc_file = open(file_name,'a')                        
                
                acc_file.write("%s \t %f \t %f \n" %(ind,precision[0],precision[1]))
                acc_file.close()
                        
                if ( (l_in == Network.no_layers-1) and (l_out == Network.no_layers-1) ):
                    first_flag2 = 0
    
        itr_T = itr_T + 1        
        #----------------------------------------------------------------------


mean_void_b = np.divide(mean_void_b,ensemble_size)
mean_void_p = np.divide(mean_void_p,ensemble_size)
mean_void_r = np.divide(mean_void_r,ensemble_size)
mean_exc = np.divide(mean_exc,ensemble_size)
mean_inh = np.divide(mean_inh,ensemble_size)
mean_void = np.divide(mean_void,ensemble_size)

std_void_b = np.divide(std_void_b,ensemble_size)
std_void_p = np.divide(std_void_p,ensemble_size)
std_void_r = np.divide(std_void_r,ensemble_size)
std_exc = np.divide(std_exc,ensemble_size)
std_inh = np.divide(std_inh,ensemble_size)
std_void = np.divide(std_void,ensemble_size)


mu_mean_void_b = mean_void_b.mean(axis = 1)
mu_mean_void_p = mean_void_p.mean(axis = 1)
mu_mean_void_r = mean_void_r.mean(axis = 1)
mu_mean_exc = mean_exc.mean(axis = 1)
mu_mean_inh = mean_inh.mean(axis = 1)
mu_mean_void = mean_void.mean(axis = 1)

mu_std_void_b = std_void_b.mean(axis = 1)
mu_std_void_p = std_void_p.mean(axis = 1)
mu_std_void_r = std_void_r.mean(axis = 1)
mu_std_exc = std_exc.mean(axis = 1)
mu_std_inh = std_inh.mean(axis = 1)
mu_std_void = std_void.mean(axis = 1)

#pdb.set_trace()

temp = np.vstack([np.array(T_range).T,mu_mean_exc.T,mu_std_exc.T])
file_name = file_name_base_results + "/Mean_var_exc_%s.txt" %file_name_ending2
np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')

temp = np.vstack([np.array(T_range).T,mu_mean_inh.T,mu_std_inh.T])
file_name = file_name_base_results + "/Mean_var_inh_%s.txt" %file_name_ending2
np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')

temp = np.vstack([np.array(T_range).T,mu_mean_void.T,mu_std_void.T])
file_name = file_name_base_results + "/Mean_var_void_%s.txt" %file_name_ending2
np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')

temp = np.vstack([np.array(T_range).T,mu_mean_void_r.T,mu_std_void_r.T])
file_name = file_name_base_results + "/Mean_var_void_recurr%s.txt" %file_name_ending2
np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')

temp = np.vstack([np.array(T_range).T,mu_mean_void_p.T,mu_std_void_p.T])
file_name = file_name_base_results + "/Mean_var_void_presyn%s.txt" %file_name_ending2
np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')

temp = np.vstack([np.array(T_range).T,mu_mean_void_b.T,mu_std_void_b.T])
file_name = file_name_base_results + "/Mean_var_void_back%s.txt" %file_name_ending2
np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')

#plt.plot(mu_mean_exc,'b'); plt.plot(mu_mean_inh,'r');plt.plot(mu_mean_void,'g');plt.plot(mu_mean_void_r,'g--'); plt.show();
#plt.plot(mu_std_exc,'b'); plt.plot(mu_std_inh,'r');plt.plot(mu_std_void,'g');plt.plot(mu_std_void_r,'g--');plt.plot(mu_std_void_b,'g-*');plt.plot(mu_std_void_p,'g-.'); plt.show();
#plt.plot(mu_mean_void_b,'g');plt.plot(mu_mean_void_p,'g--'); plt.show();



sp_prob = (in_spikes_tot>0).mean(axis=1)
sp_prob = np.reshape(sp_prob,[m,1])
W_inferred_our_tot2 = W_inferred_our_tot + np.dot(sp_prob,(1-2*sp_prob).T)

aa = W_inferred_our_tot2.mean(axis= 0)
aa = aa.reshape([1,m])
W_inferred_our_tot2 = W_inferred_our_tot2-np.dot(np.ones([m,1]),aa)


aa = abs(W_inferred_our_tot2).max(axis= 0)
aa = aa.reshape([1,m])
W_inferred_our_tot2 = np.divide(W_inferred_our_tot2,np.dot(np.ones([m,1]),aa))
W_inferred_our_tot2 = tanh(1*W_inferred_our_tot2)
plt.subplot(1, 2, 1);plt.imshow(W_inferred_our_tot2);plt.subplot(1, 2, 2);plt.imshow(np.sign(W_tot));plt.show()


thr_inh = mu_mean_inh[len(T_range)-1] + (mu_mean_void[len(T_range)-1]-mu_mean_inh[len(T_range)-1])/2.0
thr_exc1 = mu_mean_exc[len(T_range)-1] + (mu_mean_void[len(T_range)-1]-mu_mean_exc[len(T_range)-1])/2.0
thr_exc2 = mu_mean_exc[len(T_range)-1] + (mu_mean_void_r[len(T_range)-1]-mu_mean_exc[len(T_range)-1])/2.0
W_bin = 0.001*(np.multiply(W_inferred_our_tot>thr_exc1,W_inferred_our_tot<thr_exc2)).astype(int)
W_bin = W_bin- 0.005*(W_inferred_our_tot<=thr_inh).astype(int)