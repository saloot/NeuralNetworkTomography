#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.2
frac_input_neurons_default = 0.4
no_cascades_default = 8000
ensemble_size_default = 2
delay_max_default = 1.0
binary_mode_default = 4
file_name_base_result_default = "./Results/FeedForward"
inference_method_default = 0
ensemble_count_init_default = 1
no_layers_default = 3
random_delay_flag_default = 0
#==============================================================================

#================================INSTRUCTIONS==================================
help_message = "\n"
help_message = help_message + "\n"
help_message = help_message + "###################################INSTRUCTIONS################################\n"
help_message = help_message + "Here is how to use the code: you have to specify the option flag and"
help_message = help_message + "the quantity right afterwards.\nExample: -E 100 for setting a network with 100 excitatory neurons. "
help_message = help_message + "The full list of options are as follows:\n"
help_message = help_message + "-E xxx: To specify the number of excitatory neurons. Default value = %s.\n" %str(n_exc_default)
help_message = help_message + "-I xxx: To specify the number of inhibitory neurons. Default value = %s.\n" %str(n_inh_default)
help_message = help_message + "-P xxx: To specify the probabaility of having a connection between two neurons. Default value = %s.\n" %str(connection_prob_default)
help_message = help_message + "-Q xxx: To specify the fraction of stimulated input neurons. Default value = %s.\n" %str(frac_input_neurons_default)
help_message = help_message + "-T xxx: To specify the number of considered cascades. Default value = %s.\n" %str(no_cascades_default)
help_message = help_message + "-S xxx: To specify the number of generated random graphs. Default value = %s.\n" %str(ensemble_size_default)
help_message = help_message + "-D xxx: To specify the maximum delay for the neural connections in milliseconds. Default value = %s.\n" %str(delay_max_default)
help_message = help_message + "-B xxx: To specify the binarification algorithm. Default value = %s. \n" %str(binary_mode_default)
help_message = help_message + "-A xxx: To specify the folder that stores the generated data. Default value = %s. \n" %str(file_name_base_result_default)
help_message = help_message + "-F xxx: To specify the ensemble index to start simulation. Default value = %s. \n" %str(ensemble_count_init_default)
help_message = help_message + "-M xxx: To specify the method use for inference, 0 for ours, 1 for Hopfield. Default value = %s. \n" %str(inference_method_default)
help_message = help_message + "-L xxx: To specify the number of layers in the network. Default value = %s. \n" %str(no_layers_default)
help_message = help_message + "-R xxx: To specify if the delays are fixed (R=0) or random (R=1). Default value = %s. \n" %str(random_delay_flag_default)
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
from time import time
import numpy as np
import sys,getopt,os
import matplotlib.pyplot as plt
import pdb
#os.chdir('C:\Python27')
#os.chdir('/home/salavati/Desktop/Neural_Tomography')
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:A:D:F:M:")
if (input_opts):
    for opt, arg in input_opts:
        if opt == '-E':
            n_exc = int(arg)                                    # The number of excitatory neurons in the output layer
        elif opt == '-I':
                n_inh = int(arg)                                    # The number of inhibitory neurons in the output layer
        elif opt == '-P':
            connection_prob = float(arg)                        # The probability of having a link from the input to output neurons in the second layer
        elif opt == '-Q':
            frac_input_neurons = float(arg)                     # Fraction of neurons in the input layer that will be excited by a stimulus
        elif opt == '-T':
            no_cascades = int(arg)                              # Number of times we inject stimulus to the network
        elif opt == '-S':
            ensemble_size = int(arg)                            # The number of random networks that will be generated
        elif opt == '-B':
            binary_mode = int(arg)                              # Defines the method to transform the graph to binary. "1" for threshold base and "2" for sparsity based
        elif opt == '-D':
            delay_max = float(arg)                              # The maximum amount of synaptic delay in mili seconds
        elif opt == '-A':
            file_name_base_data = str(arg)                      # The folder to store results
        elif opt == '-F':
            ensemble_count_init = int(arg)                      # The index of ensemble we start with
        elif opt == '-M':
            inference_method = int(arg)                         # The inference method
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

#------------Set the Default Values if Variables are not Defines---------------
if 'frac_input_neurons' not in locals():
    frac_input_neurons = frac_input_neurons_default
    print('ATTENTION: The default value of %s for frac_input_neurons is considered.\n' %str(frac_input_neurons))

if 'no_cascades' not in locals():        
    no_cascades = no_cascades_default
    print('ATTENTION: The default value of %s for no_cascades is considered.\n' %str(no_cascades))

if 'ensemble_size' not in locals():            
    ensemble_size = ensemble_size_default
    print('ATTENTION: The default value of %s for ensemble_size is considered.\n' %str(ensemble_size))

if 'binary_mode' not in locals():
    binary_mode = binary_mode_default;
    print('ATTENTION: The default value of %s for binary_mode is considered.\n' %str(binary_mode))

if 'file_name_base_result' not in locals():
    file_name_base_result = file_name_base_result_default;
    print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_result))

if 'ensemble_count_init' not in locals():
    ensemble_count_init = ensemble_count_init_default;
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))

if 'inference_method' not in locals():
    inference_method = inference_method_default;
    print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(inference_method))
    
if 'no_layers' not in locals():
    no_layers = no_layers_default;
    print('ATTENTION: The default value of %s for no_layers is considered.\n' %str(no_layers))

if 'random_delay_flag' not in locals():
    random_delay_flag = random_delay_flag_default;
    print('ATTENTION: The default value of %s for random_delay_flag is considered.\n' %str(random_delay_flag))

if 'connection_prob' not in locals():
    connection_prob_matrix = np.zeros([no_layers,no_layers])
    for i in range(0,no_layers):
        for j in range(i,no_layers):
            l = j-i + 1
            p_temp = connection_prob_default/(math.log(j+1) + 0.5772156 + 0.85/(2*j+2))   # Using Harmonic number approximations
            p_temp = 0.001*round(1000*p_temp/float(l))
            connection_prob_matrix[i,j] = p_temp
    print('ATTENTION: The default value of %s for connection_prob is considered.\n' %str(connection_prob_matrix))
    
if 'delay_max' not in locals():
    delay_max_matrix = np.zeros([no_layers,no_layers])
    for i in range(0,no_layers):
        for j in range(i,no_layers):
            delay_max_matrix[i,j] = delay_max_default*(0.9*(j-i)+1)
    print('ATTENTION: The default value of %s for delay_max is considered.\n' %str(delay_max_matrix))

if 'n_exc_array' not in locals():
    n_exc_array = n_exc_default*np.ones([no_layers])
    print('ATTENTION: The default value of %s for n_exc is considered.\n' %str(n_exc_array))

if 'n_inh_array' not in locals():
    n_inh_array = n_inh_default*np.ones([no_layers])
    print('ATTENTION: The default value of %s for n_inh is considered.\n' %str(n_inh_array))
    
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
Neural_Spikes = {}
Inferred_Matrices_Our = {}                                                      # Initialize the inferred matrix (the matrix of belifs) using our algorithm
Inferred_Matrices_Hebian = {}                                                   # Initialize the inferred matrix (the matrix of belifs) using the hebbian algorithm
#W_inferred_our = np.zeros([n])                                                  # Initialize the inferred matrix (the matrix of belifs) using our algorithm
#W_inferred_hebian = np.zeros([n])                                               # Initialize the inferred matrix (the matrix of belifs) using the hebbian algorithm

def soft_threshold(W,thr):
    WW = np.multiply(W-thr,W>thr) + np.multiply(W+thr,W<-thr)
    return WW
#------------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)


file_name_base_data = "./Data/MultiLayerFeedForward"              # The folder to read neural data from
file_name_base_results = "./Results/MultiLayerFeedForward"        # The folder to store resutls
#delay_max = float(delay_max)                            # This is done to avoid any incompatibilities in reading the data files

T_range = range(50, no_cascades, 200)                  # The range of sample sizes considered to investigate the effect of sample size on the performance
no_samples_per_cascade = max(3.0,25*no_layers*np.max(delay_max_matrix)) # Number of samples that will be recorded
running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
sim_window = round(running_period*10*delay_max)                     # This is the number of iterations performed within each cascade
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

str_p = ''
str_d = ''
str_n_exc = ''
str_n_inh = ''
for i in range(0,no_layers):
    str_n_exc = str_n_exc + '_' + str(int(n_exc_array[i]))
    str_n_inh = str_n_inh + '_' + str(int(n_inh_array[i]))
    for j in range(i,no_layers):
        str_p = str_p + '_' + str(connection_prob_matrix[i,j])
        str_d = str_d + '_' + str(delay_max_matrix[i,j])

t_base = time()
t_base = t_base-t0
#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
B_exc_mean = np.zeros([len(T_range)])
B_inh_mean = np.zeros([len(T_range)])
B_void_mean = np.zeros([len(T_range)])
B_exc_std = np.zeros([len(T_range)])
B_inh_std = np.zeros([len(T_range)])
B_void_std = np.zeros([len(T_range)])
    
B_void_max = np.zeros([len(T_range)])
B_void_min = np.zeros([len(T_range)])
B_inh_max = np.zeros([len(T_range)])
B_exc_min = np.zeros([len(T_range)])

for ensemble_count in range(ensemble_count_init,ensemble_size):

    t0_ensemble = time()
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "L_%s" %str(int(no_layers))
    file_name_ending = file_name_ending + "_n_exc" + str_n_exc
    file_name_ending = file_name_ending + "_n_inh" + str_n_inh
    file_name_ending = file_name_ending + "_p" + str_p 
    file_name_ending = file_name_ending + "_q_%s" %str(frac_input_neurons)
    file_name_ending = file_name_ending + "_R_%s" %str(random_delay_flag)    
    file_name_ending = file_name_ending + "_d" + str_d
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)    
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------                
    #--------------------READ THE NETWORK AND SPIKE TIMINGS--------------------
    Neural_Weights = {}
    file_name_base = file_name_base_data + "/Spikes/S_times_MLFF_%s" %file_name_ending
    for l_in in range(0,no_layers):
        n_exc = n_exc_array[l_in]
        n_inh = n_inh_array[l_in]
        for l_out in range(l_in,no_layers):
            
            file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
        
            #...........................Read the Input Matrix..........................
            file_name = file_name_base_data + "/Graphs/We_MLFF_%s.txt" %file_name_ending_temp
            We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
            file_name = file_name_base_data + "/Graphs/Wi_MLFF_%s.txt" %file_name_ending_temp
            Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            
            ll = We.shape            
            if (len(ll)==1):
                We = We.reshape(n_exc,1)
                Wi = Wi.reshape(n_inh,1)
                
            W = np.vstack((We,Wi))
            
            ind = str(l_in) + str(l_out)
            Neural_Weights[ind] = list([We,Wi,W])
            #..........................................................................    
    
    #...........................Read and Sort Spikes...........................
    for l_in in range(0,no_layers+1):
        file_name = file_name_base + '_l_' + str(l_in) +'.txt'
        S_time_file = open(file_name,'r')
    
        S_times = np.genfromtxt(file_name, dtype=float, delimiter='\t')
        S_time_file.close()
        if (l_in < no_layers):
            n_exc = n_exc_array[l_in]
            n_inh = n_inh_array[l_in]
            n = n_exc + n_inh
        else:
            n = 1
            
        in_spikes = np.zeros([n,no_cascades])
        in_spikes.fill(0)
        if delay_max>0:
            recorded_spikes = np.zeros([n,no_cascades*sim_window])                      # This matrix captures the time slot in which each neuron has fired in each step
            cumulative_recorded_spikes = np.zeros([n,no_cascades*sim_window])               # This matrix takes into account the effect of neural history 
            recorded_spikes.fill(0)
            for i in range(0,no_cascades):            
                recorded_spikes[:,i*sim_window+sim_window-1] = -1
        else:
            recorded_spikes = np.zeros([n,no_cascades])
            cumulative_recorded_spikes = np.zeros([n,no_cascades*sim_window])               # This matrix takes into account the effect of neural history 
            recorded_spikes.fill(0)         
            cumulative_recorded_spikes.fill(0)
        
        recorded_spikes.fill(0)        
        s = S_times.shape
        cascade_count = 0
        for l in range(0,s[0]):
            neuron_count = int(S_times[l,0])        
            if (neuron_count == -2.0):            
                cascade_count = cascade_count + 1
            else:
                tt = round(10000*S_times[l,1])-1
                in_spikes[neuron_count,cascade_count] = S_times[l,1]
                if (tt>0):
                    recorded_spikes[neuron_count,(cascade_count)*sim_window+sim_window-1] = 0
                    
                recorded_spikes[neuron_count,(cascade_count)*sim_window+tt] = 1
                #cumulative_recorded_spikes[neuron_count,(cascade_count)*sim_window+tt+1:(cascade_count+1)*sim_window] = cumulative_recorded_spikes[neuron_count,(cascade_count)*sim_window+tt+1:(cascade_count+1)*sim_window] + np.multiply(np.ones([sim_window-tt-1]),range(1,int(sim_window-tt)))
                cumulative_recorded_spikes[neuron_count,(cascade_count)*sim_window+tt+1:(cascade_count+1)*sim_window] = np.ones([sim_window-tt-1])
                
    
        print(sum((S_times>0)))
        
        Neural_Spikes[str(l_in)] = list([recorded_spikes,cumulative_recorded_spikes,in_spikes]) #in_spikes
        
        S_times.fill(0)
        
    #..........................................................................
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
#==============================================================================



#============================INFER THE CONNECTIONS=============================
    for l_in in range(0,no_layers):
        n_exc = n_exc_array[l_in]
        n_inh = n_inh_array[l_in]
        n = n_exc + n_inh
        for l_out in range(l_in,no_layers):
            #--------------------------In-Loop Initializations--------------------------            
            
            ind = str(l_in) + str(l_out)
            
            if (l_out < no_layers-1):
                n_exc = n_exc_array[l_out+1]
                n_inh = n_inh_array[l_out+1]
                m = n_exc + n_inh
            else:
                m = 1
            temp_list = Neural_Weights[ind]
            W = temp_list[2]
            
            temp_list = Neural_Spikes[str(l_in)]
            cumulative_recorded_spikes = temp_list[1]
            in_spikes = temp_list[0]
            simple_in_spikes = temp_list[2]
            simple_in_spikes = np.sign(simple_in_spikes)
            
            temp_list = Neural_Spikes[str(l_out+1)]
            recorded_spikes = temp_list[0]
            out_spikes = temp_list[0]
            simple_out_spikes = temp_list[2]
            simple_out_spikes = np.sign(simple_out_spikes)
            
            out_spikes_s = np.sign(out_spikes)
            out_spikes_neg = -pow(-1,out_spikes_s)
            delay_max = delay_max_matrix[l_in,l_out]
            temp = delay_max*(out_spikes_s==0)/4
            out_spikes_s2 = out_spikes_s + temp
            out_spikes_s2 = (out_spikes_s2)

            #out_spikes_neg = tanh(np.multiply(out_spikes_neg,out_spikes_s2))
            L = in_spikes.sum(axis=0)
            itr = 0
    
            B_exc_mean.fill(0)
            B_inh_mean.fill(0)
            B_void_mean.fill(0)
            B_exc_std.fill(0)
            B_inh_std.fill(0)
            B_void_std.fill(0)
    
            B_void_max.fill(0)
            B_void_min.fill(0)
            B_inh_max.fill(0)
            B_exc_min.fill(0)            
            #--------------------------------------------------------------------------
            
            
            #------------------------The Generative Update Model-----------------------
            if (inference_method == 2):
                W_inferred_generative = np.zeros([m,n])
                W_inferred_generative.fill(0)
                alpha0 = 0.01
                range_tau = range(0,100)
                cost = np.zeros([len(range_tau)])
                W_inferred_generative_detailed = np.zeros([m*n,len(T_range)])
                W_inferred_generative_detailed.fill(0)
                for tau in range_tau:
                    temp = 0
                    temp_T = np.zeros([len(T_range),m*n])
                    alpha = alpha0/float(tau+1)
                    sparse_thr = 0.1/float(tau+1)
                    itr_T = 0
                    for cascade_count in range(0,no_cascades):
                        x = simple_in_spikes[:,cascade_count]
                        x = x.reshape(n,1)
                        y = simple_out_spikes[:,cascade_count]
                        y = y.reshape(m,1)
                        y_predict = 0.5*(1+np.sign(np.dot(W_inferred_generative,x)-theta))
                        y_predict = y_predict.reshape(m,1)
                        
                        
                        temp = temp + np.dot(y_predict - y,x.T)
                        
                        cost[tau] = cost[tau] + pow(y_predict - y,2)
                        if (cascade_count in T_range):
                            temp_T[itr_T,:] = temp.ravel()
                            itr_T = itr_T + 1
                    
                    if (tau>10):
                        if ( abs(cost[tau]-cost[tau-2])/float(cost[tau]) < 0.001):
                            break
                        else:                               
                            W_inferred_generative = W_inferred_generative - alpha * temp
                            temp_itr = 0
                            for T in T_range:
                                temp_update = temp_T[temp_itr]                             
                                W_inferred_generative_detailed[:,temp_itr] = W_inferred_generative_detailed[:,temp_itr] - alpha *temp_update
                                temp_itr = temp_itr + 1

                    else:
                        W_inferred_generative = W_inferred_generative - alpha * temp
                        temp_itr = 0
                        for T in T_range:
                            temp_update = temp_T[temp_itr]                             
                            W_inferred_generative_detailed[:,temp_itr] = W_inferred_generative_detailed[:,temp_itr] - alpha *temp_update
                            if (sparsity_flag):
                                W_inferred_generative_detailed[:,temp_itr] = soft_threshold(W_inferred_generative,sparse_thr)
                            #pdb.set_trace()
                            temp_itr = temp_itr + 1
                  
            #--------------------------------------------------------------------------
            itr_T = 0
            first_flag = 1
            for T in T_range:
        
                #------------------------In-Loop Initializations-----------------------
                L_temp = L[0:T]
                Delta = 1.0/float(T)
                out_spikes_neg_temp = out_spikes_neg[:,0:T]
                in_spikes_temp = in_spikes[:,0:T]
                out_spikes_temp = out_spikes_s[:,0:T]
                recorded_spikes_temp = recorded_spikes[:,0:T*sim_window]
                cumulative_spikes_temp = cumulative_recorded_spikes[:,0:T*sim_window]
                #----------------------------------------------------------------------
        
                #--------------------Construct the Belief Matrix-----------------------
                if (inference_method == 2):
                    W_inferred_our = W_inferred_generative_detailed[:,itr_T]
                    W_inferred_our = W_inferred_our.reshape([m,n])
                    W_inferred_our = W_inferred_our.T
                else:
                    W_inferred_our = np.dot(cumulative_spikes_temp,recorded_spikes_temp.T)*Delta
                
                W_inferred_hebian = np.dot((-pow(-1,np.sign(cumulative_spikes_temp))),recorded_spikes_temp.T)*Delta
                #W_inferred_our = np.dot(in_spikes_temp,out_spikes_neg_temp.T)*Delta
                #W_inferred_hebian = np.dot((-pow(-1,np.sign(in_spikes_temp))),out_spikes_neg_temp.T)*Delta
                #----------------------------------------------------------------------
        
                #------------------------Save the Belief Matrices---------------------------
                file_name_ending23 = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                file_name_ending23 = file_name_ending23 + '_I_' + str(inference_method)
                if (sparsity_flag):
                    file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
                file_name_ending2 = file_name_ending23 +"_%s" %str(T)
                file_name_ending24 = file_name_ending23 +"_T_%s" %str(no_cascades)
                
                file_name = file_name_base_results + "/Inferred_Graphs/W_MLFF_%s.txt" %file_name_ending2
                np.savetxt(file_name,W_inferred_our,'%5.3f',delimiter='\t')

                file_name = file_name_base_results + "/Inferred_Graphs/W_Hebb_MLFF_%s.txt" %file_name_ending2
                np.savetxt(file_name,W_inferred_hebian,'%5.3f',delimiter='\t')
                #---------------------------------------------------------------------------
        
                #-----------CALCULATE AND STORE THE RUNNING TIME OF THE CODE------------        
                t1 = time()                             # Capture the timer to calculate the running time per ensemble
                #print "Total simulation time was %f s" %(t1-t0_ensemble+t_base)
                file_name = file_name_base_results + "/RunningTimes/T_MLFF_%s.txt" %file_name_ending2
                running_time_file = open(file_name,'a')
                running_time_file.write("%f \n" %(t1-t0_ensemble+t_base))
                running_time_file.close()
                #----------------------------------------------------------------------
                #----------------------------------------------------------------------
                
                #----------------------------------------------------------------------
                #------------------ASSESS THE QUALITY OF THE INFERRENCE----------------
    
                #.........Caclulate the Minimum Value of the Excitatory Beliefs............                
                a = np.nonzero(W>0)
                temp = W_inferred_our[a]
                B_exc_mean[itr] = temp.mean()
                B_exc_std[itr] = temp.std()
                B_exc_min[itr] = temp.mean()
                #..........................................................................
    
                #.........Caclulate the Maximum Value of the Inhibitory Beliefs............
                a = np.nonzero(W<0)
                temp = W_inferred_our[a]
                if (len(temp) > 1):
                    B_inh_mean[itr] = temp.mean()
                    B_inh_std[itr] = temp.std()
                    B_inh_max[itr] = temp.max()
                else:
                    B_inh_mean[itr] = float('NaN') #B_inh_mean[itr-1]
                    B_inh_std[itr] = float('NaN') #B_inh_std[itr-1]
                    B_inh_max[itr] = float('NaN') #B_inh_max[itr-1]
                #..........................................................................
    
                #.......Caclulate the Minimum and Maximum Value of the Void Beliefs........
                a = np.nonzero(W==0)
                temp = W_inferred_our[a]
                B_void_mean[itr] = temp.mean()
                B_void_std[itr] = temp.std()
                B_void_max[itr] = temp.max()
                B_void_min[itr] = temp.min()
                #..........................................................................
    
    
                #.................Display and Write the Results to a File..................                   
                file_name = file_name_base_results + "/BeliefQuality/BQ_MLFF_%s.txt" %file_name_ending2
                if first_flag:
                    running_time_file = open(file_name,'a')                    
                else:
                    running_time_file = open(file_name,'w')
                running_time_file.write("%f \t %f \t %f \n" %(B_exc_mean[itr],B_inh_mean[itr],B_void_mean[itr]))
                running_time_file.close()
        
                file_name = file_name_base_results + "/BeliefQuality/BQ_MLFF_Max_Min_%s.txt" %file_name_ending2
                if first_flag:
                    running_time_file = open(file_name,'a')                    
                else:
                    running_time_file = open(file_name,'w')
                running_time_file.write("%f \t %f \t %f \t %f \n" %(B_exc_min[itr],B_void_max[itr],B_void_min[itr],B_inh_max[itr]))
                running_time_file.close()
        
        
                file_name = file_name_base_results + "/BeliefQuality/BQ_MLFF_Std_%s.txt" %file_name_ending2
                if first_flag:
                    running_time_file = open(file_name,'a')
                    first_flag = 0
                else:
                    running_time_file = open(file_name,'w')
                running_time_file.write("%f \t %f \t %f \n" %(B_exc_std[itr],B_inh_std[itr],B_void_std[itr]))
                running_time_file.close()
        
                file_name = file_name_base_results + "/Plot_Results/B_exc_mean_MLFF_%s.txt" %file_name_ending24
                running_time_file = open(file_name,'a')
                running_time_file.write("%d \t %f \t %f \n" %(T,B_exc_mean[itr],B_exc_std[itr]))
                running_time_file.close()
    
                file_name = file_name_base_results + "/Plot_Results/B_inh_mean_MLFF_%s.txt" %file_name_ending24
                running_time_file = open(file_name,'a')
                running_time_file.write("%d \t %f \t %f \n" %(T,B_inh_mean[itr],B_inh_std[itr]))
                running_time_file.close()
    
                file_name = file_name_base_results + "/Plot_Results/B_void_mean_MLFF_%s.txt" %file_name_ending24
                running_time_file = open(file_name,'a')
                running_time_file.write("%d \t %f \t %f \n" %(T,B_void_mean[itr],B_void_std[itr]))
                running_time_file.close()
                #..........................................................................
        
                itr = itr + 1
                #------------------------------------------------------------------------------
                #------------------------------------------------------------------------------
                print "B_exc_mean: %f    B_void_mean: %f    B_inh_mean: %f" %(B_exc_mean[itr-1],B_void_mean[itr-1],B_inh_mean[itr-1])
                print "B_exc_std: %f    B_void_std: %f    B_inh_std: %f\n" %(B_exc_std[itr-1],B_void_std[itr-1],B_inh_std[itr-1])
            
                itr_T = itr_T + 1
            pdb.set_trace()
            #corr = np.dot(in_spikes,out_spikes.T)
            #plt.imshow(corr)
            #plt.show()
            
    

#==================================================================================
#==================================================================================

#plt.errorbar(T_range,B_exc_mean,color='r')
plt.plot(T_range,B_exc_mean,'r')
plt.plot(T_range,B_inh_mean,'b')
plt.plot(T_range,B_void_mean,'g')
plt.show()       

plt.plot(T_range,B_exc_std,'r')
plt.plot(T_range,B_inh_std,'b')
plt.plot(T_range,B_void_std,'g')
plt.show()       
continue

plt.plot(T_range,B_exc_min,'r')
plt.plot(T_range,B_void_max,'g')
plt.plot(T_range,B_void_min,'g--')
plt.plot(T_range,B_inh_max,'b')
plt.show()       

T = 100
file_name_ending25 = file_name_ending2 +"_%s" %str(T)
TT = range(0,T)
for l in range (0,no_layers+1):
    
    temp_list = Neural_Spikes[str(l)]
    temp = temp_list[2]
    Firing_Pat = np.sign(temp[0,0:T])
    Firing_Pat = Firing_Pat.tolist()
    Firing_Pat = zip(TT,Firing_Pat)
    file_name = file_name_base_results + "/Plot_Results/Firing_Pattern_N_1_L_%s_%s.txt" %(str(l),file_name_ending25)
    np.savetxt(file_name,Firing_Pat,'%1.0f',delimiter='\t',newline="\n")
