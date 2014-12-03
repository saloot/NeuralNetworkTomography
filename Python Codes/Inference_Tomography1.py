#=======================DEFAULT VALUES FOR THE VARIABLES=======================
FRAC_STIMULATED_NEURONS_DEFAULT = 0.4
NO_STIMUL_ROUNDS_DEFAULT = 1
ENSEMBLE_SIZE_DEFAULT = 1
FILE_NAME_BASE_DATA_DEFAULT = "./Data"
FILE_NAME_BASE_RESULT_DEFAULT = "./Results"
ENSEMBLE_COUNT_INIT_DEFAULT = 0
INFERENCE_METHOD_DEFAULT = 3
BINARY_MODE_DEFAULT = 4
SPARSITY_FLAG_DEFAULT = 0
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
from auxiliary_functions import inference_alg_per_layer
from auxiliary_functions import read_spikes
from auxiliary_functions import calculate_belief_quality
import Neurons_and_Networks
reload(Neurons_and_Networks)
from Neurons_and_Networks import NeuralNet
from Neurons_and_Networks import *

#import brian
#reload(brian)
#from brian import *
#spikequeue.SpikeQueue.reinit
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

plot_flag = 0

os.system('clear')                                              # Clear the commandline window
t0 = time()                                                # Initialize the timer
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:G:")
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
        elif opt == '-B':
            binary_mode = int(arg)                              # Defines the method to transform the graph to binary. "1" for threshold base and "2" for sparsity based                        
        elif opt == '-M':
            inference_method = int(arg)                         # The inference method
        elif opt == '-G':
            sparsity_flag = int(arg)                            # The flag that determines if sparsity should be observed during inference
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

#------------------------------------------------------------------------------

#--------------------------Initialize the Network------------------------------
Network = NeuralNet(no_layers,n_exc_array,n_inh_array,connection_prob_matrix,delay_max_matrix,random_delay_flag,'')
#------------------------------------------------------------------------------    

#---------------------Initialize Simulation Variables--------------------------
no_samples_per_cascade = max(3.0,25*Network.no_layers*np.max(Network.delay_max_matrix)) # Number of samples that will be recorded
network_type = 'R'
if network_type == 'F':
    running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
else:
    running_period = (10*no_samples_per_cascade/10.0)  # Total running time in mili seconds

sim_window = round(1+running_period*10)                     # This is the number of iterations performed within each cascade

theta = 0.005                              # The update threshold of the neurons in the network
T_range = range(350, no_stimul_rounds, 300)                  # The range of sample sizes considered to investigate the effect of sample size on the performance
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
if not os.path.isdir(file_name_base_results+'/BeliefQuality'):    
    temp = file_name_base_results + '/BeliefQuality'
    os.makedirs(temp)
if not os.path.isdir(file_name_base_results+'/Plot_Results'):    
    temp = file_name_base_results + '/Plot_Results'
    os.makedirs(temp)    
#------------------------------------------------------------------------------

t_base = time()
t_base = t_base-t0
#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
for ensemble_count in range(ensemble_count_init,ensemble_size):

    t0_ensemble = time()
    
    #--------------------------READ THE NETWORK--------------------------------
    Network.read_weights(ensemble_count,file_name_base_data)
    #--------------------------------------------------------------------------
    
    #--------------------------Read and Sort Spikes----------------------------
    file_name_base = file_name_base_data + "/Spikes/S_times_%s" %Network.file_name_ending + "_q_%s" %str(frac_stimulated_neurons)
    Neural_Spikes = read_spikes(file_name_base,Network.no_layers,Network.n_exc_array,Network.n_inh_array,no_stimul_rounds,sim_window)    
    #--------------------------------------------------------------------------
    
#==============================================================================



#============================INFER THE CONNECTIONS=============================
    for l_out in range(0,Network.no_layers):
        
        n_exc = Network.n_exc_array[l_out]
        n_inh = Network.n_inh_array[l_out]
        m = n_exc + n_inh
            
        #--------------------------In-Loop Initializations--------------------------
        B_exc_mean = np.zeros([m,len(T_range)])
        B_inh_mean = np.zeros([m,len(T_range)])
        B_void_mean = np.zeros([m,len(T_range)])
    
        B_void_max = np.zeros([m,len(T_range)])
        B_void_min = np.zeros([m,len(T_range)])
        B_inh_max = np.zeros([m,len(T_range)])
        B_exc_min = np.zeros([m,len(T_range)])
    
        temp_list = Neural_Spikes[str(l_out)]            
        out_spikes = (temp_list[2])
        recorded_spikes = temp_list[0]
        
        in_spikes = []
        n = 0
        Belief_Qualities = {}
        
        for l_in in range(0,l_out+1):
            temp_list = Neural_Spikes[str(l_in)]
            #pdb.set_trace()
            if len(in_spikes):
                in_spikes = np.concatenate([in_spikes,temp_list[2]])
            else:
                in_spikes = temp_list[2]
            cumulative_recorded_spikes = temp_list[1]
            
            Belief_Qualities['B_exc_mean_'+str(l_in)] = np.zeros([m,len(T_range)])
            Belief_Qualities['B_inh_mean_'+str(l_in)] = np.zeros([m,len(T_range)])
            Belief_Qualities['B_void_mean_'+str(l_in)] = np.zeros([m,len(T_range)])

            Belief_Qualities['B_void_max_'+str(l_in)] = np.zeros([m,len(T_range)])
            Belief_Qualities['B_void_min_'+str(l_in)] = np.zeros([m,len(T_range)])
            Belief_Qualities['B_inh_max_'+str(l_in)] = np.zeros([m,len(T_range)])
            Belief_Qualities['B_exc_min_'+str(l_in)] = np.zeros([m,len(T_range)])
            
        alpha0 = 0.00001
        sparse_thr0 = 0.0001
        #--------------------------------------------------------------------------
                  
        #--------------------------------------------------------------------------           
        itr = 0
        first_flag = 1
        
        for T in T_range:
                    
            #------------------------In-Loop Initializations-----------------------
            in_spikes_temp = in_spikes[:,0:T]
            out_spikes_temp = out_spikes[:,0:T]
            recorded_spikes_temp = recorded_spikes[:,0:T*sim_window]
            cumulative_spikes_temp = cumulative_recorded_spikes[:,0:T*sim_window]
            #----------------------------------------------------------------------
        
            #--------------------Construct the Belief Matrix-----------------------
            if (inference_method == 3):
                inferece_params = [alpha0,sparse_thr0,sparsity_flag,theta,W_binary,sim_window]
                #W_inferred_our_tot,cost,no_decision_plus,no_decision_minus = inference_alg_per_layer(in_spikes_temp,out_spikes_temp,inference_method,inferece_params)
                binned_spikes = bin_spikes(recorded_spikes_temp,0.001,sim_window)
                W_inferred_our_tot,cost,no_decision_plus,no_decision_minus = inference_alg_per_layer(binned_spikes,recorded_spikes_temp,inference_method,inferece_params)
                print "\a"
                #pdb.set_trace()
                #W_inferred_our_tot = W_inferred_our_tot.T
            elif (inference_method == 0):
                inferece_params = [1]
                W_inferred_our_tot = inference_alg_per_layer(cumulative_spikes_temp,recorded_spikes_temp,inference_method,inferece_params)
                
            #.....The Hebbian Algorithm.......
            else:
                inferece_params = [1]
                W_inferred_our_tot = inference_alg_per_layer((-pow(-1,np.sign(cumulative_spikes_temp))),recorded_spikes_temp,inference_method,inferece_params)
            #----------------------------------------------------------------------
            
            #-----------CALCULATE AND STORE THE RUNNING TIME OF THE CODE------------        
            #t1 = time()                             # Capture the timer to calculate the running time per ensemble
            #print "Total simulation time was %f s" %(t1-t0_ensemble+t_base)
            #file_name = file_name_base_results + "/RunningTimes/T_%s.txt" %file_name_ending2
            #running_time_file = open(file_name,'a')
            #running_time_file.write("%f \n" %(t1-t0_ensemble+t_base))
            #running_time_file.close()
            #----------------------------------------------------------------------
                
            n_so_far = 0
            #if l_out == 2:
            #    if T>500:
            #        pdb.set_trace()
            for l_in in range(0,l_out+1):    
                n_exc = Network.n_exc_array[l_in]
                n_inh = Network.n_inh_array[l_in]
                this_n = n_exc + n_inh
                
            
                ind = str(l_in) + str(l_out)
                temp_list = Network.Neural_Connections[ind]
                W = temp_list[0]
                W_inferred_our = W_inferred_our_tot[n_so_far:n_so_far+this_n,:]
                #W_inferred_our = W_inferred_our.T
                n_so_far = n_so_far + this_n
                
                #------------------------Save the Belief Matrices---------------------------
                file_name_ending23 = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                file_name_ending23 = file_name_ending23 + '_I_' + str(inference_method)
                if (sparsity_flag):
                    file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
                file_name_ending2 = file_name_ending23 +"_T_%s" %str(T)
                file_name_ending24 = file_name_ending23 +"_T_%s" %str(no_stimul_rounds)
                
                if (W_inferred_our.max()>0):
                    W_inferred_our = W_inferred_our/float(W_inferred_our.max())
                
                if (W_inferred_our.min()<0):
                    W_inferred_our = W_inferred_our/float(-W_inferred_our.min())
                file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending2
                np.savetxt(file_name,W_inferred_our,'%1.5f',delimiter='\t')
                #---------------------------------------------------------------------------
        

                
                
                #------------------------------------------------------------------------------
                Mean_Beliefs,Worst_Beliefs = calculate_belief_quality(W_inferred_our,W)

                
                Belief_Qualities['B_exc_mean_'+str(l_in)][:,itr] = Mean_Beliefs[:,0]
                Belief_Qualities['B_void_mean_'+str(l_in)][:,itr] = Mean_Beliefs[:,1]
                Belief_Qualities['B_inh_mean_'+str(l_in)][:,itr] = Mean_Beliefs[:,2]
                
                Belief_Qualities['B_exc_min_'+str(l_in)][:,itr] = Worst_Beliefs[:,0]
                Belief_Qualities['B_void_max_'+str(l_in)][:,itr] = Worst_Beliefs[:,1]
                Belief_Qualities['B_void_min_'+str(l_in)][:,itr] = Worst_Beliefs[:,2]
                Belief_Qualities['B_inh_max_'+str(l_in)][:,itr] = Worst_Beliefs[:,3]
                
                
                                
                if plot_flag:
                    bar_width = 0.2
                    index = np.arange(int(n))
                    plt.bar(index, B_exc_mean[:,itr], bar_width,color='r')
                    plt.bar(index+bar_width, B_void_mean[:,itr], bar_width,color='g')
                    plt.bar(index+2*bar_width, B_inh_mean[:,itr], bar_width,color='b')
                    plt.show()
                    
                    plt.bar(index, B_exc_min[:,itr], bar_width,color='r')
                    plt.bar(index+bar_width, B_void_max[:,itr], bar_width,color='g')
                    plt.bar(index+2*bar_width, B_void_min[:,itr], bar_width,color='y')
                    plt.bar(index+3*bar_width, B_inh_max[:,itr], bar_width,color='b')
                    plt.show()                    
                

            itr = itr + 1
                
                #------------------------------------------------------------------------------
                #print "B_exc_mean: %f    B_void_mean: %f    B_inh_mean: %f" %(B_exc_mean[itr-1],B_void_mean[itr-1],B_inh_mean[itr-1])
                #print "B_exc_std: %f    B_void_std: %f    B_inh_std: %f\n" %(B_exc_std[itr-1],B_void_std[itr-1],B_inh_std[itr-1])
            
                
            #-------Calculate the Minimum T Required for Belief Separation--------
            min_T_belief_separ = float('NaN')*np.ones([m])
            for ii in range(0,int(m)):
                itr_T = 0
                min_flag = 0
                for T in T_range:
                    if ( (B_exc_min[ii,itr_T] > B_void_max[ii,itr_T]) and (B_void_min[ii,itr_T] > B_inh_max[ii,itr_T])):
                        if (min_flag == 0):
                            min_T_belief_separ[ii] = T
                            min_flag = 1
                            
                    else:
                        min_flag = 0
                        min_T_belief_separ[ii] = float('NaN')
                    itr_T = itr_T + 1
            #---------------------------------------------------------------------
            
            
            #-------------------Write the Results to the File---------------------                           
            file_name = file_name_base_results + "/BeliefQuality/BQ_Mean_Exc_%s.txt" %file_name_ending24
            np.savetxt(file_name,np.vstack([T_range,B_exc_mean]).T,'%5.3f',delimiter='\t',newline='\n')
            
            file_name = file_name_base_results + "/BeliefQuality/BQ_Mean_Inh_%s.txt" %file_name_ending24
            np.savetxt(file_name,np.vstack([T_range,B_inh_mean]).T,'%5.3f',delimiter='\t',newline='\n')
            
            file_name = file_name_base_results + "/BeliefQuality/BQ_Mean_Void_%s.txt" %file_name_ending24
            np.savetxt(file_name,np.vstack([T_range,B_void_mean]).T,'%5.3f',delimiter='\t',newline='\n')
            
            file_name = file_name_base_results + "/BeliefQuality/BQ_Worst_Exc_Min_%s.txt" %file_name_ending24
            np.savetxt(file_name,np.vstack([T_range,B_exc_min]).T,'%5.3f',delimiter='\t',newline='\n')
            
            file_name = file_name_base_results + "/BeliefQuality/BQ_Worst_Void_Max_%s.txt" %file_name_ending24
            np.savetxt(file_name,np.vstack([T_range,B_void_max]).T,'%5.3f',delimiter='\t',newline='\n')
                        
            file_name = file_name_base_results + "/BeliefQuality/BQ_Worst_Void_Min_%s.txt" %file_name_ending24
            np.savetxt(file_name,np.vstack([T_range,B_void_min]).T,'%5.3f',delimiter='\t',newline='\n')
            
            file_name = file_name_base_results + "/BeliefQuality/BQ_Worst_Inh_Max_%s.txt" %file_name_ending24
            np.savetxt(file_name,np.vstack([T_range,B_inh_max]).T,'%5.3f',delimiter='\t',newline='\n')
            
            file_name = file_name_base_results + "/BeliefQuality/Min_T_Separation%s.txt" %file_name_ending24
            np.savetxt(file_name,min_T_belief_separ,'%5.1f',delimiter='\t',newline='\n')
            #---------------------------------------------------------------------
                            
            #----------------------------Plot Results-----------------------------
            #if (l_out == 1):
            #    pdb.set_trace()
            if plot_flag:
                selected_neuron = 0
                plt.plot(T_range,B_exc_mean[selected_neuron,:],'r--')
                plt.plot(T_range,B_inh_mean[selected_neuron,:],'b--')
                plt.plot(T_range,B_void_mean[selected_neuron,:],'g--')
                plt.show()       
                    
                plt.plot(T_range,B_exc_min[selected_neuron,:],'r')
                plt.plot(T_range,B_void_max[selected_neuron,:],'g')
                plt.plot(T_range,B_void_min[selected_neuron,:],'g--')
                plt.plot(T_range,B_inh_max[selected_neuron,:],'b')
                plt.show()       
                #plt.errorbar(T_range,B_exc_mean,color='r')
            #---------------------------------------------------------------------
    

#==================================================================================
#==================================================================================
