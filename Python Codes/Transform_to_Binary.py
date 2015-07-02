#=======================DEFAULT VALUES FOR THE VARIABLES=======================
FRAC_STIMULATED_NEURONS_DEFAULT = 0.4
NO_STIMUL_ROUNDS_DEFAULT = 2000
ENSEMBLE_SIZE_DEFAULT = 1
FILE_NAME_BASE_DATA_DEFAULT = "./Data"
FILE_NAME_BASE_RESULT_DEFAULT = "./Results"
ENSEMBLE_COUNT_INIT_DEFAULT = 0
BINARY_MODE_DEFAULT = 4
INFERENCE_METHOD_DEFAULT = 3
SPARSITY_FLAG_DEFAULT = 0
GENERATE_DATA_MODE_DEFAULT = 'R'
INFERENCE_ITR_MAX_DEFAULT = 1
WE_KNOW_LOCATION_DEFAULT = 'Y'
PRE_SYNAPTIC_NEURON_DEFAULT = 'A'
DELAY_KNOWN_DEFAULT = 'N'
#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
import time
import numpy as np
import sys,getopt,os
from scipy.cluster.vq import kmeans,whiten,kmeans2,vq

import Neurons_and_Networks
reload(Neurons_and_Networks)
from Neurons_and_Networks import NeuralNet
from Neurons_and_Networks import *

from auxiliary_functions import beliefs_to_binary
from auxiliary_functions import calucate_accuracy
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
help_message = help_message + "-A xxx: To specify the folder that stores the generated data. Default value = %s. \n" %str(FILE_NAME_BASE_RESULT_DEFAULT)
help_message = help_message + "-F xxx: To specify the ensemble index to start simulation. Default value = %s. \n" %str(ENSEMBLE_COUNT_INIT_DEFAULT)
help_message = help_message + "-L xxx: To specify the number of layers in the network. Default value = %s. \n" %str(NO_LAYERS_DEFAULT)
help_message = help_message + "-R xxx: To specify if the delays are fixed (R=0) or random (R=1). Default value = %s. \n" %str(RANDOM_DELAY_FLAG_DEFAULT)
help_message = help_message + "-B xxx: To specify the binarification algorithm. Default value = %s. \n" %str(BINARY_MODE_DEFAULT)
help_message = help_message + "-M xxx: To specify the method use for inference, 0 for ours, 1 for Hopfield. Default value = %s. \n" %str(INFERENCE_METHOD_DEFAULT)
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================

#================================INITIALIZATIONS===============================
n_exc_array = None; n_inh_array= None; connection_prob_matrix = None
random_delay_flag = None; no_layers = None; delay_max_matrix = None

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time.time()                                                     # Initialize the timer

input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:J:K:C:")
if (input_opts):
    for opt, arg in input_opts:        
        if opt == '-Q':
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
        elif opt == '-B':
            binary_mode = int(arg)                              # Defines the method to transform the graph to binary. "1" for threshold base and "2" for sparsity based                        
        elif opt == '-M':
            inference_method = int(arg)                         # The inference method
        elif opt == '-G':
            generate_data_mode = str(arg)                            # The flag that determines if sparsity should be observed during inference        
        elif opt == '-Y':
            sparsity_flag = int(arg)                            # The flag that determines if sparsity should be observed during inference
        elif opt == '-X':
            infer_itr_max = int(arg)                            # The flag that determines if sparsity should be observed during inference            
        elif opt == '-K':
            we_know_location = str(arg)                         # The flag that determines if we know the location of neurons (with respect to each other) (Y/N)
        elif opt == '-C': 
            pre_synaptic_method = str(arg)                      # The flag that determines if all previous-layers neurons count as  pre-synaptic (A/O)
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

if 'file_name_base_results' not in locals():
    file_name_base_results = FILE_NAME_BASE_RESULT_DEFAULT;
    print('ATTENTION: The default value of %s for file_name_base_results is considered.\n' %str(file_name_base_results))
    
if 'ensemble_count_init' not in locals():
    ensemble_count_init = ENSEMBLE_COUNT_INIT_DEFAULT;
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))
    
if 'binary_mode' not in locals():
    binary_mode = BINARY_MODE_DEFAULT;
    print('ATTENTION: The default value of %s for binary_mode is considered.\n' %str(binary_mode))

if 'inference_method' not in locals():
    inference_method = INFERENCE_METHOD_DEFAULT;
    print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(inference_method))

if 'sparsity_flag' not in locals():
    sparsity_flag = SPARSITY_FLAG_DEFAULT;
    print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(sparsity_flag))

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

if 'delay_known_flag' not in locals():
    delay_known_flag = DELAY_KNOWN_DEFAULT
    print('ATTENTION: The default value of %s for delay_known_flag is considered.\n' %str(delay_known_flag))
#------------------------------------------------------------------------------

#--------------------------Initialize the Network------------------------------
#Network = NeuralNet(no_layers,n_exc_array,n_inh_array,connection_prob_matrix,delay_max_matrix,random_delay_flag,'')
Network = NeuralNet(None,None,None,None,None,None,None, 'command_line',input_opts,args)
#------------------------------------------------------------------------------    

#--------------------------Initialize Other Variables--------------------------
theta = 0.005                                               # The update threshold of the neurons in the network

#if (generate_data_mode == 'F'):
#    T_range = range(350, no_stimul_rounds, 300)                 # The range of sample sizes considered to investigate the effect of sample size on the performance
#else:
#    T_range = range(350, int(running_period*10), 300)
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if NEcessary---------------
if not os.path.isdir(file_name_base_results):
    os.makedirs(file_name_base_results)

temp = file_name_base_results + '/Accuracies'
if not os.path.isdir(temp):
    os.makedirs(temp)
    
temp = file_name_base_results + '/RunningTimes'
if not os.path.isdir(temp):
    os.makedirs(temp)

temp = file_name_base_results + '/Inferred_Graphs'
if not os.path.isdir(temp):
    os.makedirs(temp)
#------------------------------------------------------------------------------


#-----------------------Initialize Simulation Variables------------------------
adj_fact_exc = 0.75 # 0.125 # 0.25 # 0.375 # 0.5 # 0.625 # 0.75 # 0.875 # 1 # 1.125 # 1.25 # 1.375 # 1.5 # 1.625 # 1.75    # 0--> Prec     Inf--> Recall
adj_fact_inh = 0.5 # 0.125 # 0.25 # 0.375 # 0.5 # 0.625 # 0.75 # 0.875 # 1 # 1.125 # 1.25 # 1.375 # 1.5 # 1.625 # 1.75
#------------------------------------------------------------------------------
                
t_base = time.time()
t_base = t_base-t0
#==============================================================================

T_range = [100,2779,5458,8140,10820,13500]

#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
for ensemble_count in range(ensemble_count_init,ensemble_size):    
    t0_ensemble = time.time()
    first_flag2 = 1
    
    #--------------------------READ THE NETWORK--------------------------------
    Network.read_weights(ensemble_count,file_name_base_data)
    #--------------------------------------------------------------------------        
    
    #-------------------------IN-LOOP INITIALIZATIONS--------------------------
    recall_tot = {}
    prec_tot = {}    
    for l_in in range(0,Network.no_layers):
        for l_out in range(l_in,Network.no_layers):            
            if we_know_location == 'Y':
                ind = str(l_in) + '_' + str(l_out)        
                recall_tot[ind] = np.zeros([len(T_range),3])
                prec_tot[ind] = np.zeros([len(T_range),3])
            else:
                #............Construct the Concatenated Weight Matrix..............            
                W_tot = []            
                n_so_far = 0
                for l_in in range(0,Network.no_layers):
                    n_exc = Network.n_exc_array[l_in]
                    n_inh = Network.n_inh_array[l_in]
                    n = n_exc + n_inh
                    
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
    
    if we_know_location != 'Y':
        recall_tot = np.zeros([len(T_range),3])
        prec_tot = np.zeros([len(T_range),3])
    #--------------------------------------------------------------------------
     
    #---------------------------CALCULATE ACCURACY-----------------------------       
    itr_T = 0
    for T in T_range:        
        if we_know_location == 'Y':
            
            
            print '=====RESULTS IN LAYER: l_in=%s and l_out=%s========' %(l_in,l_out)
        
            for l_in in range(0,Network.no_layers):
                file_name_ending23 = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                n_exc = Network.n_exc_array[l_in]
                n_inh = Network.n_inh_array[l_in]
                for l_out in range(l_in+1,Network.no_layers):
                    ind = str(l_in) + str(l_out)
                    temp_list = Network.Neural_Connections[ind]
                    W = temp_list[0]
                
                    #-------------------------Read the Belief Matrices-------------------------        
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
                    W_inferred = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    #--------------------------------------------------------------------------
                    
                    #-----------------Calculate the Binary Matrix From Beliefs-----------------
                    n,m = W.shape
                    fixed_entries = np.zeros([n,m])
                    W_inferred = W_inferred[:,0:m]
                    params = [adj_fact_exc,adj_fact_inh,fixed_entries]
                    W_inferred = W_inferred/float(abs(W_inferred).max())
                    W_inferred = W_inferred + np.random.rand(n,m)/100000
                    W_inferred = whiten(W_inferred)
                    W_binary,centroids = beliefs_to_binary(binary_mode,10*W_inferred,params,0)                    
                    #--------------------------------------------------------------------------
                    
                    #--------------------------Store the Binary Matrices-----------------------
                    file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
                    file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                    file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
                
                    file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2
                    np.savetxt(file_name,W_binary,'%d',delimiter='\t',newline='\n')
                
                    file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_%s.txt" %file_name_ending2                
                    ww = W_inferred.ravel()
                    ww = np.vstack([ww,np.zeros([len(ww)])])
                    np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')

                    if (binary_mode == 4):
                        file_name = file_name_base_results + "/Inferred_Graphs/Centroids_%s.txt" %file_name_ending2
                        centroids = np.vstack([centroids,np.zeros([3])])
                        np.savetxt(file_name,centroids,'%f',delimiter='\t')
                    #--------------------------------------------------------------------------
                    
                    #---------Calculate and Display Recall & Precision for Our Method----------    
                    recal,precision = calucate_accuracy(W_binary,W)
                    ind = str(l_in) + '_' + str(l_out)
                    temp1 = recall_tot[ind]
                    temp1[itr_T,:] = recal
                    recall_tot[ind] = temp1
                    
                    temp1 = prec_tot[ind]
                    temp1[itr_T,:] = precision
                    prec_tot[ind] = temp1
                
                    print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
                    print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
                    print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
                    print '\n'
                    #--------------------------------------------------------------------------
                    
              
                    
        else:
            
            #-------------------------Read the Belief Matrices-------------------------
            file_name_ending23 = Network.file_name_ending 
            file_name_ending23 = file_name_ending23 + '_I_' + str(inference_method)
            file_name_ending23 = file_name_ending23 + '_Loc_' + we_know_location
            file_name_ending23 = file_name_ending23 + '_Pre_' + pre_synaptic_method
            file_name_ending23 = file_name_ending23 + '_G_' + generate_data_mode
            file_name_ending23 = file_name_ending23 + '_X_' + str(infer_itr_max)
            file_name_ending23 = file_name_ending23 + '_Q_' + str(frac_stimulated_neurons)
            if (sparsity_flag):
                file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
            file_name_ending2 = file_name_ending23 +"_T_%s" %str(T)
            n,m = W_tot.shape
            fixed_entries = np.zeros([n,m])
            file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending2
            W_inferred = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            W_inferred = W_inferred[:,0:m]
            #--------------------------------------------------------------------------

            #-----------------Calculate the Binary Matrix From Beliefs-----------------
            n,m = W_tot.shape
            fixed_entries = np.zeros([n,m])
            params = [adj_fact_exc,adj_fact_inh,fixed_entries]
            W_inferred = W_inferred/float(abs(W_inferred).max())
            W_inferred = W_inferred + np.random.rand(n,m)/100000
            W_inferred = whiten(W_inferred)
            if binary_mode == 10:
                p_exc = 0.32
                p_inh = 0.08
                W_inferred_our_tot = W_inferred
                gamma_pos_range = np.array(range(0,int(1000*W_inferred_our_tot.max())))/1000.0
                gamma_neg_range = -np.array(range(0,abs(int(1000*W_inferred_our_tot.min()))))/1000.0
                W_exc = np.zeros([n,n])
                W_inh = np.zeros([n,n])
                for j in range(0,n):
                    itr_e = 0
                    for gamma_pos in gamma_pos_range:
                        W_exc[:,j] = (W_inferred_our_tot[:,j]>gamma_pos).astype(int)
                        if sum(W_exc[:,j])/float(n) < p_exc:
                            break
                        itr_e = itr_e + 1
                
                    if (itr_e < len(gamma_pos_range)):
                        gamma_pos = gamma_pos_range[itr_e]
                        W_exc[:,j] = (W_inferred_our_tot[:,j]>gamma_pos).astype(int)
                    else:
                        W_exc[:,j] = 0 * (W_inferred_our_tot[:,j]>gamma_pos).astype(int)
        
                    itr_e = 0
        
                    for gamma_neg in gamma_neg_range:
                        W_inh[:,j] = (W_inferred_our_tot[:,j] < gamma_neg).astype(int)
                        if sum(W_inh[:,j])/float(n) < p_inh:
                            break
                        itr_e = itr_e + 1
        
                    if (itr_e < len(gamma_neg_range)):
                        gamma_neg = gamma_neg_range[itr_e]
                        W_inh[:,j] = (W_inferred_our_tot[:,j] < gamma_neg).astype(int)
                    else:
                        W_inh[:,j] = 0 * (W_inferred_our_tot[:,j] < gamma_neg).astype(int)
    
    
                W_binary = W_exc - W_inh
                centroids = [0,0,0]
            else:
                W_binary,centroids = beliefs_to_binary(binary_mode,10*W_inferred,params,0)
            #--------------------------------------------------------------------------
        
            #--------------------------Store the Binary Matrices-----------------------
            file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
            file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
            file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
                
            file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2
            np.savetxt(file_name,W_binary,'%d',delimiter='\t',newline='\n')
        
                
            file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_%s.txt" %file_name_ending2                
            ww = W_inferred.ravel()
            ww = np.vstack([ww,np.zeros([len(ww)])])
            np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')

            if (binary_mode == 4):
                file_name = file_name_base_results + "/Inferred_Graphs/Centroids_%s.txt" %file_name_ending2
                centroids = np.vstack([centroids,np.zeros([3])])
                np.savetxt(file_name,centroids,'%f',delimiter='\t')
            #--------------------------------------------------------------------------
                
            #---------Calculate and Display Recall & Precision for Our Method----------    
            recal,precision = calucate_accuracy(W_binary,W_tot)
            recall_tot[itr_T,:] = recal
            prec_tot[itr_T,:] = precision
                
            print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
            print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
            print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
            print '\n'
            #--------------------------------------------------------------------------
                
        itr_T = itr_T + 1
    #======================================================================================
        
        
    #==================================SAVE THE RESULTS====================================
    T_range = np.divide(T_range,1000.0).astype(int)
    if we_know_location == 'Y':    
        for l_in in range(0,Network.no_layers):
            n_exc = Network.n_exc_array[l_in]
            n_inh = Network.n_inh_array[l_in]
            for l_out in range(l_in+1,Network.no_layers):
                recal = recall_tot[ind]
                precs = prec_tot[ind]
                
                temp_ending = file_name_ending2.replace("_T_%s" %str(T),'')
                file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %temp_ending
                np.savetxt(file_name,np.vstack([T_range,recal.T]).T,'%f',delimiter='\t',newline='\n')
        
                file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %temp_ending
                np.savetxt(file_name,np.vstack([T_range,precs.T]).T,'%f',delimiter='\t',newline='\n')
        
                #file_name_ending2 = file_name_ending2.replace('_l_' + str(l_in) + '_to_' + str(l_out),'')        
                #file_name = file_name_base_results + "/Accuracies/Rec_Layers_Segregated_%s.txt" %file_name_ending2
                #if (first_flag2):
                #    acc_file = open(file_name,'w')
                #else:
                #    acc_file = open(file_name,'a')
            
                #acc_file.write("%s \t %f \t %f \n" %(ind,recal[itr_T,0],recal[itr_T,1]))
                #acc_file.close()
        
                #file_name = file_name_base_results + "/Accuracies/Prec_Layers_Segregated_%s.txt" %file_name_ending2

                #if (first_flag2):
                #    acc_file = open(file_name,'w')
                #else:
                #    acc_file = open(file_name,'a')
                #    if ( (l_in == Network.no_layers-1) and (l_out == Network.no_layers-1) ):
                #        first_flag2 = 0
            
                #acc_file.write("%s \t %f \t %f \n" %(ind,prec_tot[itr_T,0],prec_tot[itr_T,1]))
                #acc_file.close()
    else:
        temp_ending = file_name_ending2.replace("_T_%s" %str(T),'')
        file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %temp_ending
        np.savetxt(file_name,np.vstack([T_range,recall_tot.T]).T,'%f',delimiter='\t',newline='\n')
        
        file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %temp_ending
        np.savetxt(file_name,np.vstack([T_range,prec_tot.T]).T,'%f',delimiter='\t',newline='\n')
        #==================================================================================
            #raw_input("Press a key to continue...")    
        