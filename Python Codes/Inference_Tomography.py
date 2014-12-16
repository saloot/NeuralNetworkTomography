#=======================DEFAULT VALUES FOR THE VARIABLES=======================
FRAC_STIMULATED_NEURONS_DEFAULT = 0.4
NO_STIMUL_ROUNDS_DEFAULT = 2000
ENSEMBLE_SIZE_DEFAULT = 1
FILE_NAME_BASE_DATA_DEFAULT = "./Data"
FILE_NAME_BASE_RESULT_DEFAULT = "./Results"
ENSEMBLE_COUNT_INIT_DEFAULT = 0
INFERENCE_METHOD_DEFAULT = 3
BINARY_MODE_DEFAULT = 4
SPARSITY_FLAG_DEFAULT = 0
GENERATE_DATA_MODE_DEFAULT = 'F'
INFERENCE_ITR_MAX_DEFAULT = 1
WE_KNOW_LOCATION_DEFAULT = 'Y'
PRE_SYNAPTIC_NEURON_DEFAULT = 'A'
#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import os
import sys,getopt,os
import matplotlib.pyplot as plt
import pdb

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
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:G:X:Y:K:C:")
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
    running_period = (100*no_samples_per_cascade/10.0)  # Total running time in mili seconds

sim_window = round(1+running_period*10)                     # This is the number of iterations performed within each cascade

theta = 0.005                                               # The update threshold of the neurons in the network


if (generate_data_mode == 'F'):
    T_range = range(200, no_stimul_rounds, 150)                 # The range of sample sizes considered to investigate the effect of sample size on the performance
else:
    T_range = range(350, int(running_period*10), 300)
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

t_base = time()
t_base = t_base-t0
alpha0 = 0.000001
sparse_thr0 = 0.0001
adj_fact_exc = 0.75
adj_fact_inh = 0.5
#-------------------------Initialize Inference Parameters----------------------

#............................Correlation-Bases Approach........................
if (inference_method == 0):
    inferece_params = [1,theta]           
#..............................................................................

#............................Perceptron-Based Approach.........................
elif (inference_method == 3) or (inference_method == 2):
    #~~~~~We Know the Location of Neurons + Stimulate and Fire~~~~~
    # Current version
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
    #~~~We Know the Location of Neurons + Continuous Stimulation~~~
    # Just give the integrated version to the inference algorithm, i.e.: sum(in_spikes[stimul_span],out_spike)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
    inferece_params = [alpha0,sparse_thr0,sparsity_flag,theta,20]
#..............................................................................

#.............................Event-Based Approach.............................
elif (inference_method == 5):
    W_binary = []
    inferece_params = [alpha0,sparse_thr0,sparsity_flag,theta,sim_window]
#..............................................................................
            
#...............................CCF-Based Approach.............................
elif (inference_method == 4):
    d_window = 10
    inferece_params = [d_window]            
#..............................................................................
        
#............................Hebbian-Based Approach............................
else:
    inferece_params = [1]
    #W_inferred_our_tot = inference_alg_per_layer((-pow(-1,np.sign(cumulative_spikes_temp))),recorded_spikes_temp,inference_method,inferece_params)
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
        Neural_Spikes = read_spikes(file_name_base,Network.no_layers,Network.n_exc_array,Network.n_inh_array,['c',no_stimul_rounds,sim_window])
    else:
        Neural_Spikes = read_spikes(file_name_base,Network.no_layers,Network.n_exc_array,Network.n_inh_array,['u',int(running_period*10)])
    #--------------------------------------------------------------------------
    
    
    
#==============================================================================

    

#============================INFER THE CONNECTIONS=============================
    
            
    #--------------------------In-Loop Initializations-------------------------
    if (generate_data_mode == 'R'):
        #------------------Pre Process the Spikes if Necessary-----------------
        Processed_Neural_Spikes = {}
        for l in range(0,Network.no_layers):                
            temp_list = Neural_Spikes[str(l)]
            spikes = temp_list[2]
            n,T = spikes.shape
            out_spikes = np.zeros([n,T])
            for t in range(0,T):
                out_spikes[:,t] = sum(spikes[:,max(t-10,0):t],axis = 1)
                
            Processed_Neural_Spikes[str(l)] = (out_spikes>0).astype(int)        
    #--------------------------------------------------------------------------
    
    
    
    first_flag2 = 1    
    #--------------------------------------------------------------------------           
    
    for T in T_range:
        
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
                
                #.....If ALL previous Layers Count as Pre-synaptic Neurons.....        
                if pre_synaptic_method.lower() == 'a':
                    in_spikes = []
                    

                    #~~~~~~~~~~~~~~Concatenate Pre-synaptic Spikes~~~~~~~~~~~~~
                    n_tot = 0
                    for l_in in range(0,l_out):
                        temp_list = Neural_Spikes[str(l_in)]
                        n_exc = Network.n_exc_array[l_in]
                        n_inh = Network.n_inh_array[l_in]
                        n_tot = n_tot + n_exc + n_inh                
                        if len(in_spikes):
                            in_spikes_temp = temp_list[2]
                            in_spikes_temp = in_spikes_temp[:,0:T]
                            in_spikes = np.concatenate([in_spikes,in_spikes_temp])
                        else:
                            in_spikes = temp_list[2]
                            in_spikes = in_spikes[:,0:T]
                    
                    in_spikes_orig = in_spikes
                    if (generate_data_mode != 'R') and (inference_method != 4):
                        in_spikes = (in_spikes>0).astype(int)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~~~~~~
                    n = n_tot   
                    W_estimated = np.zeros([n,m])
                    fixed_entries = np.zeros([n,m])
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~~~~Perfrom Inference~~~~~~~~~~~~~~~~~~~~~
                    ind = str(l_in) + str(l_out);temp_list = Network.Neural_Connections[ind];W = temp_list[0]
                    for infer_itr in range(0,infer_itr_max):
                        W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes,out_spikes,inference_method,inferece_params,W_estimated,1)
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
                        
                        W_estimated,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,1.25*(1+infer_itr/7.5),2.5*(1+infer_itr/15.0),],0)
                        
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
                            W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes,out_spikes,inference_method,inferece_params,W_estimated,1)
                            W_inferred_our_tot = W_inferred_our_tot/(abs(W_inferred_our_tot).max())
                            W_estimated,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_ind],0)#
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
                    if (sparsity_flag):
                        file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
                    file_name_ending2 = file_name_ending23 +"_T_%s" %str(T)                        

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
                        W_bin,centroids = beliefs_to_binary(binary_mode,1000*W_inferred_our_tot,params,0)
                    #pdb.set_trace()
                    file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
                    file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                    file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
                
                    file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2
                    np.savetxt(file_name,W_bin,'%d',delimiter='\t',newline='\n')
        
                
                    file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_%s.txt" %file_name_ending2                
                    ww = W_inferred_our.ravel()
                    ww = np.vstack([ww,np.zeros([len(ww)])])
                    np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')

                    if (binary_mode == 4):
                        file_name = file_name_base_results + "/Inferred_Graphs/Centroids_%s.txt" %file_name_ending2
                        centroids = np.vstack([centroids,np.zeros([3])])
                        np.savetxt(file_name,centroids,'%f',delimiter='\t')
                #..............................................................
                
                #....................Calculate the Accuracy....................
                    recal,precision = calucate_accuracy(W_bin,W)
                    
                    print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
                    print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
                    print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
                    print '\n'
                #..............................................................
                    
                    
                #.............Calculate Spike Prediction Accuracy..............
                    if 1: #(T == T_range[len(T_range)-1]):
                        Network_in = {}
                        Network_in['n_in'] = n
                        Network_in['n_out'] = m
                        Network_in['d_max'] = 1
                        sps_pred = np.zeros([m,T])
                        W_bin = np.multiply(W_bin>0,0.001*np.ones([n,m])) + np.multiply(W_bin<0,-0.005*np.ones([n,m]))                        
                        
                        temp_W = W #W_bin
                        temp_W = temp_W.reshape([n,m])
                        Network_in['W'] = temp_W
                        Network_in['D'] = np.sign(abs(temp_W))/10000.0                    
                        
                        for t in range(0,T):
                            sps_in = in_spikes_orig[:,t]
                            stimulated_neurons = np.nonzero(sps_in)
                            stimulated_neurons = stimulated_neurons[0]                            
                            stimulation_times = sps_in[stimulated_neurons] * 1000.0
                            Neural_Connections_Out,out_spike = verify_neural_activity(Network,Network_in,running_period,frac_stimulated_neurons,stimulated_neurons,stimulation_times)
                            sps_pred[:,t] = out_spike                                
                            
                        
                        print sum(abs(np.sign(out_spikes) - np.sign(sps_pred) ))
                                            
                        file_name = file_name_base_results + "/Verified_Spikes/Predicted_Spikes_%s.txt" %file_name_ending2
                        save_matrix = np.zeros([10*min(T,100),3])                        
                        itr_mat = 0
                        for ik in range(0,10):
                            for ij in range(0,min(T,100)):
                                val = 1 + np.sign(out_spikes[ik,ij]) - np.sign(sps_pred[ik,ij])
                                
                                save_matrix[itr_mat,:] = [ij,ik,val]
                                itr_mat = itr_mat + 1
                        
                        np.savetxt(file_name,save_matrix,'%3.5f',delimiter='\t')
                        
                        file_name = file_name_base_results + "/Verified_Spikes/Average_Mismatch_%s.txt" %file_name_ending2
                        avg_mismatch = sum(abs(np.sign(out_spikes) - np.sign(sps_pred) ),axis = 1)
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
            in_spikes_tot = []
            out_spikes_tot = []
            for l_in in range(0,Network.no_layers):
                n_exc = Network.n_exc_array[l_in]
                n_inh = Network.n_inh_array[l_in]
                n = n_exc + n_inh
                
                temp_list = Neural_Spikes[str(l_in)]            
                spikes = (temp_list[2])
                #spikes = (spikes>0).astype(int)
                if len(out_spikes_tot):
                    out_spikes_tot = np.vstack([out_spikes_tot,spikes])
                else:
                    out_spikes_tot = spikes
                
                    
                W_temp = []
                for l_out in range(0,Network.no_layers):
                    n_exc = Network.n_exc_array[l_out]
                    n_inh = Network.n_inh_array[l_out]
                    m = n_exc + n_inh
                        
                    if (l_out < l_in):
                        if len(W_temp):
                            W_temp = np.hstack([W_temp,np.zeros([n,m])])
                        else:
                            W_temp = np.zeros([n,m])
                    else:                                        
                        ind = str(l_in) + str(l_out);
                        temp_list = Network.Neural_Connections[ind];
                        W = temp_list[0]
                        
                        if len(W_temp):
                            W_temp = np.hstack([W_temp,W])
                        else:
                            W_temp = W
                
                
                
                if len(W_tot):
                    W_tot = np.vstack([W_tot,W_temp])
                else:
                    W_tot = W_temp
            #..................................................................
            
            #......................In-Loop Initializations.....................
            W = W_tot
            m,n = W.shape
            W_estimated = np.zeros([n,m])
            fixed_entries = np.zeros([n,m])
            out_spikes_tot = out_spikes_tot[:,0:T]
            in_spikes_tot = out_spikes_tot            
            #..................................................................
            
                        
            #........................Perfrom Inference.........................
            for infer_itr in range(0,infer_itr_max):
                
                W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes_tot,out_spikes_tot,inference_method,inferece_params,W_estimated,0)
                W_tmp = abs(np.ma.masked_array(W_inferred_our_tot,mask= (W_inferred_our_tot==float('inf')).astype(int)))
                
                if (inference_method == 4):
                    for ii in range (0,m):
                        for jj in range(0,n):
                            if (W_inferred_our_tot[ii,jj]==float('inf')):
                                W_inferred_our_tot[ii,jj] = 1*W_tmp.max()
                            
                W_temp = np.ma.masked_array(W_inferred_our_tot,mask= fixed_entries)
                
                W_estimated,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,1.25*(1+infer_itr/7.5),2500*(1+infer_itr/15.0),],0)
                
                
                W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes_tot,out_spikes_tot,inference_method,inferece_params,W_estimated,0)
                W_tmp = abs(np.ma.masked_array(W_inferred_our_tot,mask= (W_inferred_our_tot==float('inf')).astype(int)))
                
                if (inference_method == 4):
                    for ii in range (0,m):
                        for jj in range(0,n):
                            if (W_inferred_our_tot[ii,jj]==float('inf')):
                                W_inferred_our_tot[ii,jj] = 1*W_tmp.max()
                            
                W_temp = np.ma.masked_array(W_inferred_our_tot,mask= fixed_entries)
                
                W_estimated_inh,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,12500*(1+infer_itr/7.5),1.75*(1+infer_itr/15.0),],0)
                
                for ii in range (0,m):
                    for jj in range(0,n):
                        if not np.isnan(W_estimated_inh[ii,jj]):
                            W_estimated[ii,jj] = W_estimated_inh[ii,jj]
                        
                        if np.isnan(W_estimated[ii,jj]):
                            W_estimated[ii,jj] = 0
                
                
                fixed_entries = 1-isnan(W_estimated).astype(int)
                W_inferred_our_tot,cost,Updated_Vals = inference_alg_per_layer(in_spikes_tot,out_spikes_tot,inference_method,inferece_params,W_estimated,0)
                
                if norm(1-fixed_entries) == 0:
                    break
            
            
            
            W_temp = W_temp/float(max_val)/1000.0
            W_inferred_our = W_temp.data
            #..................................................................
            
            
            #...................Save the Belief Matrices...................                                                
            file_name_ending23 = Network.file_name_ending + '_I_' + str(inference_method)
            file_name_ending23 = file_name_ending23 + '_Loc_' + we_know_location
            file_name_ending23 = file_name_ending23 + '_Pre_' + pre_synaptic_method
            file_name_ending23 = file_name_ending23 + '_G_' + generate_data_mode
            file_name_ending23 = file_name_ending23 + '_X_' + str(infer_itr_max)
            if (sparsity_flag):
                file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
            file_name_ending2 = file_name_ending23 +"_T_%s" %str(T)

            file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending2
            np.savetxt(file_name,W_inferred_our,'%1.5f',delimiter='\t')
            #..............................................................
                
                
            #.....................Transform to Binary......................
            #fixed_entries = 1- np.sign(Updated_Vals)
            pdb.set_trace()
            params = [adj_fact_exc,adj_fact_inh,fixed_entries]
            if binary_mode == 7:
                W_bin,centroids = beliefs_to_binary(7,W_inferred_our_tot,[fixed_entries,0.5*(1+infer_itr/4.0),1.0*(1+infer_itr/2.0)],0)                    
                for i in range(0,m):
                    for j in range(0,n):
                        if isnan(W_bin[j,i]):
                            W_bin[j,i] = 0
            else:
                W_inferred_our_tot = W_inferred_our_tot
                W_bin,centroids = beliefs_to_binary(binary_mode,1000*W_inferred_our_tot,params,0)
            
            file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
            file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
            file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
                
            file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2
            np.savetxt(file_name,W_bin,'%d',delimiter='\t',newline='\n')
        
                
            file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_%s.txt" %file_name_ending2                
            ww = W_inferred_our.ravel()
            ww = np.vstack([ww,np.zeros([len(ww)])])
            np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')

            if (binary_mode == 4):
                file_name = file_name_base_results + "/Inferred_Graphs/Centroids_%s.txt" %file_name_ending2
                centroids = np.vstack([centroids,np.zeros([3])])
                np.savetxt(file_name,centroids,'%f',delimiter='\t')
            #..............................................................
                
            #....................Calculate the Accuracy....................
            recal,precision = calucate_accuracy(W_bin,W)
                    
            print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
            print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
            print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
            print '\n'
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
