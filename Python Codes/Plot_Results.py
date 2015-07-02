#=======================DEFAULT VALUES FOR THE VARIABLES=======================
FRAC_STIMULATED_NEURONS_DEFAULT = 0.4
NO_STIMUL_ROUNDS_DEFAULT = 2000
ENSEMBLE_SIZE_DEFAULT = 1
FILE_NAME_BASE_DATA_DEFAULT = "./Data"
FILE_NAME_BASE_RESULT_DEFAULT = "./Results"
FILE_NAME_BASE_PLOT_DEFAULT = "./Plot_Results"
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

os.system('clear')                                              # Clear the commandline window
t0 = time()                                                # Initialize the timer
actual_W_flag = 0                                  # If, it means that we are interested in reading the verified spikes for the actual weight matrix
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

if 'file_name_base_result' not in locals():
    file_name_base_result = FILE_NAME_BASE_RESULT_DEFAULT;
    print('ATTENTION: The default value of %s for file_name_base_result is considered.\n' %str(file_name_base_result))

if 'file_name_base_plot' not in locals():
    file_name_base_plot = FILE_NAME_BASE_PLOT_DEFAULT;
    print('ATTENTION: The default value of %s for file_name_base_plot is considered.\n' %str(file_name_base_plot))

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
#------------------------------------------------------------------------------

#--------------------------Initialize the Network------------------------------
Network = NeuralNet(no_layers,n_exc_array,n_inh_array,connection_prob_matrix,delay_max_matrix,random_delay_flag,'')
#------------------------------------------------------------------------------


#--------------------------Initialize Other Variables--------------------------
q_range = [0.2,0.3,0.4]                                 # The range of stimulated fraction of neurons considered in simulations
a_range = np.arange(0.0,2,0.125)

no_samples_per_cascade = max(3.0,25*Network.no_layers*np.max(Network.delay_max_matrix)) # Number of samples that will be recorded

if generate_data_mode == 'F':
    running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
else:
    running_period = (10*no_samples_per_cascade/10.0)  # Total running time in mili seconds

sim_window = round(1+running_period*10)                     # This is the number of iterations performed within each cascade

if (generate_data_mode == 'F'):
    T_range = range(200, no_stimul_rounds, 150)                 # The range of sample sizes considered to investigate the effect of sample size on the performance
else:
    T_range = range(350, int(running_period*10), 300)

    
considered_var = 'T'                                    # The variable against which we plot the performance

if (considered_var == 'T'):
    var_range = T_range
elif (considered_var == 'q'):
    var_range = q_range                                 # The variable against which we plot the performance
elif (considered_var == 'a'):
    var_range = a_range
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if Necessary---------------
if not os.path.isdir(file_name_base_plot):
    os.makedirs(file_name_base_plot)
#------------------------------------------------------------------------------


#---------------------Initialize Simulation Variables--------------------------
det_Prec_exc = np.zeros([ensemble_size-ensemble_count_init,len(var_range)])                        # Detailed precision of the algorithm (per ensemble) for excitatory connections
det_Prec_inh = np.zeros([ensemble_size-ensemble_count_init,len(var_range)])                        # Detailed precision of the algorithm (per ensemble) for inhibitory connections
det_Prec_zero  = np.zeros([ensemble_size-ensemble_count_init,len(var_range)])                      # Detailed precision of the algorithm (per ensemble) for "void" connections
det_Spike_Acc = np.zeros([ensemble_size-ensemble_count_init,len(var_range)])                           # Detailed spike reproduction accuracy

Prec_exc = np.zeros([len(var_range)])                                          # Precision of the algorithm for excitatory connections
Prec_inh = np.zeros([len(var_range)])                                          # Precision of the algorithm for inhibitory connections
Prec_zero  = np.zeros([len(var_range)])                                        # Precision of the algorithm for "void" connections
Prec_total  = np.zeros([len(var_range)])                                       # Precision of the algorithm averaged over all connections

std_Prec_exc = np.zeros([len(var_range)])                                      # Standard Deviation of precision of the algorithm for excitatory connections
std_Prec_inh = np.zeros([len(var_range)])                                      # Standard Deviation of precision of the algorithm for inhibitory connections
std_Prec_zero  = np.zeros([len(var_range)])                                    # Standard Deviation of precision of the algorithm for "void" connections
std_Prec_total  = np.zeros([len(var_range)])                                   # Standard Deviation of precision of the algorithm averaged over all connections

det_Rec_exc = np.zeros([ensemble_size-ensemble_count_init,len(var_range)])                         # Detailed recall of the algorithm for (per ensemble) excitatory connections
det_Rec_inh = np.zeros([ensemble_size-ensemble_count_init,len(var_range)])                         # Detailed recall of the algorithm for (per ensemble) inhibitory connections
det_Rec_zero  = np.zeros([ensemble_size-ensemble_count_init,len(var_range)])                       # Detailed recall of the algorithm for (per ensemble) "void" connections

Rec_exc = np.zeros([len(var_range)])                                           # Recall of the algorithm for excitatory connections
Rec_inh = np.zeros([len(var_range)])                                           # Recall of the algorithm for inhibitory connections
Rec_zero  = np.zeros([len(var_range)])                                         # Recall of the algorithm for "void" connections
Rec_total= np.zeros([len(var_range)])                                          # Recall of the algorithm averaged over all connections

std_Rec_exc = np.zeros([len(var_range)])                                       # Standard Deviation of recall of the algorithm for excitatory connections
std_Rec_inh = np.zeros([len(var_range)])                                       # Standard Deviation of recall of the algorithm for excitatory connections
std_Rec_zero  = np.zeros([len(var_range)])                                     # Standard Deviation of recall of the algorithm for excitatory connections
std_Rec_total= np.zeros([len(var_range)])                                      # Standard Deviation of recall of the algorithm averaged over all connections
#------------------------------------------------------------------------------

#==============================================================================



#=============================READ THE RESULTS=================================
itr = 0
adj_fact_exc = 0.75 #1 #1.25 #1.75 
adj_fact_inh = 0.5 #1 #1.25 #1.75

Network.read_weights(0,file_name_base_data)
file_name_ending1 = Network.file_name_ending
file_name_ending1 = file_name_ending1[0:len(file_name_ending1)-2]

for l_in in range(0,Network.no_layers):
    n_exc = Network.n_exc_array[l_in]
    n_inh = Network.n_inh_array[l_in]
    n = n_exc + n_inh
    for l_out in range(l_in,Network.no_layers):
        p = Network.connection_prob_matrix[l_in,l_out]
        p_exc = p * n_exc/float(n)
        p_inh = p * n_inh/float(n)
        
        for ensemble_count in range(ensemble_count_init,ensemble_size):
            
            if actual_W_flag == 0:
                file_name_ending2 = file_name_ending1 + "_%s" %str(ensemble_count)
                if we_know_location == 'Y':
                    file_name_ending2 = file_name_ending2 + '_l_' + str(l_in) + '_to_' + str(l_out)
                file_name_ending2 = file_name_ending2 + '_I_' + str(inference_method)
                file_name_ending2 = file_name_ending2 + '_Loc_' + we_know_location
                file_name_ending2 = file_name_ending2 + '_Pre_' + pre_synaptic_method
                file_name_ending2 = file_name_ending2 + '_G_' + generate_data_mode
                file_name_ending2 = file_name_ending2 + '_X_' + str(infer_itr_max)
                file_name_ending2 = file_name_ending2 + '_Q_' + str(frac_stimulated_neurons)
                if (sparsity_flag):
                    file_name_ending2 = file_name_ending2 + '_S_' + str(sparsity_flag)
            
            
                file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
                file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
            
                #------------------------------Read the Precisions-----------------------------
                file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %file_name_ending2            
                precision_tot = np.genfromtxt(file_name, dtype='float', delimiter='\t')
                
                var_range = precision_tot[:,0]
                det_Prec_exc[ensemble_count-ensemble_count_init,:] = (precision_tot[:,1]).T
                det_Prec_inh[ensemble_count-ensemble_count_init,:] = (precision_tot[:,2]).T
                det_Prec_zero[ensemble_count-ensemble_count_init,:] = (precision_tot[:,3]).T
                #------------------------------------------------------------------------------ 
                    
                #--------------------------------Read the Recall-------------------------------
                file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %file_name_ending2
                recall_tot = np.genfromtxt(file_name, dtype='float', delimiter='\t')
                
                det_Rec_exc[ensemble_count-ensemble_count_init,:] = (recall_tot[:,1]).T
                det_Rec_inh[ensemble_count-ensemble_count_init,:] = (recall_tot[:,2]).T
                det_Rec_zero[ensemble_count-ensemble_count_init,:] = (recall_tot[:,3]).T
                #------------------------------------------------------------------------------
            
            
            #======================READ SPIKE REPRODUCTION RESULTS=========================
                if verify_flag:
                    file_name = file_name_base_results + "/Verified_Spikes/Spike_Acc_per_T_%s.txt" %file_name_ending2
                    read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
                    det_Spike_Acc[ensemble_count-ensemble_count_init,:] = read_vals[:,1].T
            else:
                file_name_ending2 = file_name_ending1 + "_%s" %str(ensemble_count)
                if we_know_location == 'Y':
                    file_name_ending2 = file_name_ending2 + '_l_' + str(l_in) + '_to_' + str(l_out)                
                file_name_ending2 = file_name_ending2 + '_Loc_' + we_know_location
                file_name_ending2 = file_name_ending2 + '_Pre_' + pre_synaptic_method
                file_name_ending2 = file_name_ending2 + '_G_' + generate_data_mode                            
                if verify_flag:
                    file_name = file_name_base_results + "/Verified_Spikes/ACT_Spike_Acc_per_T_%s.txt" %file_name_ending2
                    read_vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
                    det_Spike_Acc[ensemble_count-ensemble_count_init,:] = read_vals[:,1].T
            #==============================================================================


        #===================CALCULATE THE MEAN AND VAR OF RESULTS======================

        #-----------------------------Calculating the Mean-----------------------------
        Prec_exc = det_Prec_exc.mean(axis=0)
        Prec_inh = det_Prec_inh.mean(axis=0)
        Prec_zero = det_Prec_zero.mean(axis=0)
        Rec_exc = det_Rec_exc.mean(axis=0)
        Rec_inh = det_Rec_inh.mean(axis=0)
        Rec_zero = det_Rec_zero.mean(axis=0)

        Prec_total = p_exc*Prec_exc+p_inh*Prec_inh+(1-p)*Prec_zero
        Rec_total = p_exc * Rec_exc+p_inh*Rec_inh+(1-p)*Rec_zero

        det_Prec_total = p_exc*det_Prec_exc+p_inh*det_Prec_inh+(1-p)*det_Prec_zero
        det_Rec_total = p_exc * det_Rec_exc+p_inh*det_Rec_inh+(1-p)*det_Rec_zero
        
        Spike_Acc = det_Spike_Acc.mean(axis=0)
        #------------------------------------------------------------------------------


        #---------------------Calculating the Standard Deviation-----------------------
        std_Prec_exc = det_Prec_exc.std(axis=0)
        std_Prec_inh = det_Prec_inh.std(axis=0)
        std_Prec_zero = det_Prec_zero.std(axis=0)
        std_Prec_total = det_Prec_total.std(axis=0)

        std_Rec_exc = det_Rec_exc.std(axis=0)
        std_Rec_inh = det_Rec_inh.std(axis=0)
        std_Rec_zero = det_Rec_zero.std(axis=0)
        std_Rec_total = det_Rec_total.std(axis=0)
        
        std_Spike_Acc = det_Spike_Acc.std(axis=0)
        #------------------------------------------------------------------------------
        #==============================================================================

        if 1:
            pdb.set_trace()
            #==============================PLOT THE RESULTS================================
            plt.plot(var_range,Prec_exc,'r');plt.plot(var_range,Prec_inh,'b');plt.plot(var_range,Prec_zero,'g');plt.plot(var_range,Rec_exc,'r--');plt.plot(var_range,Rec_inh,'b--');plt.plot(var_range,Rec_zero,'g--');plt.show()
            #==============================================================================


        #=========================WRITE THE RESULTS TO THE FILE=========================
        
        #--------------------------Construct Prpoper File Names-------------------------
        if (considered_var == 'T'):
            file_name_ending = "Effect_T"
        elif (considered_var == 'q'):        
            file_name_ending = "Effect_q_"
        elif (considered_var == 'a'):        
            file_name_ending = "Effect_a_"
        
        if we_know_location == 'Y':
            file_name_ending = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)+'_'
        
        file_name_ending = file_name_ending + Network.file_name_ending
        
        if (considered_var != 'q'):
            file_name_ending = file_name_ending + "_q_%s" %str(frac_stimulated_neurons)
        file_name_ending = file_name_ending + "_%s" %str(ensemble_size)
        
        if (considered_var != 'T'):
            file_name_ending = file_name_ending + "_T_%s" %str(no_stimul_rounds)    

        if actual_W_flag == 0:
            file_name_ending = file_name_ending + '_I_' + str(inference_method)
            if (considered_var != 'a'):
                file_name_ending = file_name_ending + "_%s" %str(adj_fact_exc)
                file_name_ending = file_name_ending +"_%s" %str(adj_fact_inh)
            file_name_ending = file_name_ending + "_B_%s" %str(binary_mode)
            if (sparsity_flag):
                file_name_ending = file_name_ending + "_S_%s" %str(sparsity_flag)
            file_name_ending = file_name_ending + '_X_' + str(infer_itr_max)
            
        file_name_ending = file_name_ending + '_Loc_' + we_know_location
        file_name_ending = file_name_ending + '_Pre_' + pre_synaptic_method
        file_name_ending = file_name_ending + '_G_' + generate_data_mode
        
        #-------------------------------------------------------------------------------

        #-----------------------Write the Results to the File---------------------------
        if actual_W_flag == 0:
            temp = np.vstack([var_range,Prec_exc,std_Prec_exc])
            file_name = file_name_base_plot + "/Prec_exc_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

            temp = np.vstack([var_range,Prec_inh,std_Prec_inh])
            file_name = file_name_base_plot + "/Prec_inh_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')
    
            temp = np.vstack([var_range,Prec_zero,std_Prec_zero])
            file_name = file_name_base_plot + "/Prec_zero_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

            temp = np.vstack([var_range,Prec_total,std_Prec_total])
            file_name = file_name_base_plot + "/Prec_total_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

            temp = np.vstack([var_range,Rec_exc,std_Rec_exc])
            file_name = file_name_base_plot + "/Reca_exc_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

            temp = np.vstack([var_range,Rec_inh,std_Rec_inh])
            file_name = file_name_base_plot + "/Reca_inh_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

            temp = np.vstack([var_range,Rec_zero,std_Rec_zero])
            file_name = file_name_base_plot + "/Reca_zero_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

            temp = np.vstack([var_range,Rec_total,std_Rec_total])
            file_name = file_name_base_plot + "/Reca_total_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

            temp = np.vstack([Rec_exc,Prec_exc,std_Rec_exc,std_Prec_exc])
            file_name = file_name_base_plot + "/Reca_vs_Prec_exc_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')

            temp = np.vstack([Rec_inh,Prec_inh,std_Rec_inh,std_Prec_inh])
            file_name = file_name_base_plot + "/Reca_vs_Prec_inh_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')
    
            temp = np.vstack([Rec_zero,Prec_zero,std_Rec_zero,std_Prec_zero])
            file_name = file_name_base_plot + "/Reca_vs_Prec_zero_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')

            temp = np.vstack([Rec_total,Prec_total,std_Rec_total,std_Prec_total])
            file_name = file_name_base_plot + "/Reca_vs_Prec_total_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')

            temp = np.vstack([var_range,np.multiply(Prec_exc,Rec_exc)])
            file_name = file_name_base_plot + "/Prec_mult_Rec_exc_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

            temp = np.vstack([var_range,np.multiply(Prec_inh,Rec_inh)])
            file_name = file_name_base_plot + "/Prec_mult_Rec_inh_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')
            
            temp = np.vstack([var_range,Spike_Acc,std_Spike_Acc])
            file_name = file_name_base_plot + "/Spike_Acc_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')
        else:
            temp = np.vstack([var_range,Spike_Acc])
            file_name = file_name_base_plot + "/ACT_Spike_Acc_%s.txt" %file_name_ending
            np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------



#==============================================================================
