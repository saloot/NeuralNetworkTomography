#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.2
frac_input_neurons_default = 0.45
no_cascades_default = 1000
ensemble_size_default = 1
delay_max_default = 1.0
binary_mode_default = 4
file_name_base_result_default = "./Results/FeedForward"
inference_method_default = 0
ensemble_count_init_default = 0
no_layers_default = 4
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
from brian import *
from time import time
import numpy as np
import sys,getopt,os
import matplotlib.pyplot as plt
import pdb
import auxiliary_functions
reload(auxiliary_functions)
from auxiliary_functions import verify_neural_activity
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
#------------------------------------------------------------------------------


#--------------------------Initialize Other Variables--------------------------
theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)


file_name_base_data = "./Data/MultiLayerFeedForward"              # The folder to read neural data from
file_name_base_results = "./Results/MultiLayerFeedForward"        # The folder to store resutls
#delay_max = float(delay_max)                            # This is done to avoid any incompatibilities in reading the data files

T_range = [50,650,1450] #range(650, 700, 200)                  # The range of sample sizes considered to investigate the effect of sample size on the performance
no_samples_per_cascade = max(3.0,25*no_layers*np.max(delay_max_matrix)) # Number of samples that will be recorded
running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds
sim_window = round(running_period*10*np.max(delay_max_matrix))                     # This is the number of iterations performed within each cascade
#------------------------------------------------------------------------------


#------------------Create the Necessary Directories if NEcessary---------------
if not os.path.isdir(file_name_base_results):
    os.makedirs(file_name_base_results)

temp = file_name_base_results + '/VerifiedSpikes'
if not os.path.isdir(temp):        
    os.makedirs(temp)
#------------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
Neural_Spikes = {}

no_samples_per_cascade = max(3.0,12*no_layers*np.max(delay_max_matrix)) # Number of samples that will be recorded
running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds

file_name_base_data = "./Data/MultiLayerFeedForward"              # The folder to read neural data from
file_name_base_results = "./Results/MultiLayerFeedForward"        # The folder to store resutls
adj_fact_exc = 1.0
adj_fact_inh = 0.5
binary_mode = 4
norm_spike_layers = np.zeros([no_layers+1,len(T_range)])
norm_spike_layers_det = np.zeros([no_layers+1,len(T_range)])
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
#------------------------------------------------------------------------------

#----------------------------------Neural Model--------------------------------
tau=10*ms
tau_e=2*ms # AMPA synapse
eqs='''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''

neural_model_eq = list([eqs,tau,tau_e])
#------------------------------------------------------------------------------

#==============================================================================




for ensemble_count in range(ensemble_count_init,ensemble_size):

#============================GENERATE THE NETWORK==============================
    norm_spike_layers_det.fill(0)
    
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
    
    W_flag = 1
    Neural_Connections = {}
    Neural_Connections['n_exc'] = n_exc_array
    Neural_Connections['n_inh'] = n_inh_array
    
    Neural_Weights = {}
    Neural_Weights['n_exc'] = n_exc_array
    Neural_Weights['n_inh'] = n_inh_array
        #...........................Read and Sort Spikes...........................
    file_name_base = file_name_base_data + "/Spikes/S_times_MLFF_%s" %file_name_ending
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
        
        
        s = S_times.shape
        cascade_count = 0
        for l in range(0,s[0]):
            neuron_count = int(S_times[l,0])        
            if (neuron_count == -2.0):            
                cascade_count = cascade_count + 1
            else:
                
                in_spikes[neuron_count,cascade_count] = S_times[l,1]
                
    
        print(sum((S_times>0)))
        
        Neural_Spikes[str(l_in)] = list([in_spikes]) #in_spikes


    stimulus = Neural_Spikes['0']
    stimulus = stimulus[0]    
    itr_T = 0
    for T in T_range:
        for l_in in range(0,no_layers):
            n_exc = n_exc_array[l_in]
            n_inh = n_inh_array[l_in]
            n = n_exc + n_inh
            for l_out in range(l_in,no_layers):
                
                file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                file_name_ending2 = file_name_ending_temp + '_I_' + str(inference_method)
                file_name_ending2 = file_name_ending2 +"_%s" %str(T)
                file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
                file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
                
                
                
                #file_name = file_name_base_results + "/Inferred_Graphs/W_MLFF_%s.txt" %file_name_ending2
                file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_Our_MLFF_Cascades_Delay_%s.txt" %file_name_ending2
            
                W = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                if (l_out == no_layers-1):
                    W = W.reshape(n,1)        
            
                delay_max = delay_max_matrix[l_in,l_out]
                ind = str(l_in) + str(l_out)
                W = W/1000.0
                Neural_Connections[ind] = list([W,'',delay_max])
            
                
                
        
                #...........................Read the Original Matrix..........................
                file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                file_name = file_name_base_data + "/Graphs/We_MLFF_%s.txt" %file_name_ending_temp
                We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
                file_name = file_name_base_data + "/Graphs/Wi_MLFF_%s.txt" %file_name_ending_temp
                Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            
                ll = We.shape            
                if (len(ll)==1):
                    We = We.reshape(n_exc,1)
                    Wi = Wi.reshape(n_inh,1)
                
                W_orig = np.vstack((We,Wi))
            
                ind = str(l_in) + str(l_out)
                Neural_Weights[ind] = list([W_orig,'',delay_max])

        
        #------------------------------------------------------------------------------
    
                
    
        #-------------------Read The stimulations----------------------
        file_name_base = file_name_base_results + "/VerifiedSpikes/S_times_MLFF_%s" %file_name_ending2
        for l_in in range(0,no_layers+1):
            file_name = file_name_base + '_l_' + str(l_in) +'.txt'
            S_time_file = open(file_name,'w')
            S_time_file.close()
            
        for cascade_count in range(0,T):
        
        
            input_stimulus = np.nonzero(stimulus[:,cascade_count])
            input_stimulus = input_stimulus[0]
            input_stimulus = input_stimulus.astype(int)
            #--------------------------Generate Activity---------------------------
            Neural_Connections_Out,Spikes_list = verify_neural_activity('F',Neural_Connections,running_period,file_name_base,neural_model_eq,input_stimulus,cascade_count,no_layers)
            #----------------------------------------------------------------------
        
            #------------------------Check for Any Errors--------------------------            
            for l_in in range(0,no_layers):
                for l_out in range(l_in,no_layers):
                    ind = str(l_in) + str(l_out)
                
                    temp_list1 = Neural_Connections[ind]
                    #temp_list1 = Neural_Weights[ind]
                    temp_list2 = Neural_Connections_Out[ind]
                
                    if (norm(temp_list1[0]-temp_list2[0])):
                        print('something is wrong with W!')
                        pdb.set_trace()
                        sys.exit()
                
            
            verified_spikes = Spikes_list['l_1']                        
            temp_list = Neural_Spikes['0']
            temp_list = temp_list[0]
            correct_spikes = temp_list[:,cascade_count]
            if (norm(correct_spikes-verified_spikes)):
                print('something is wrong with s!')
                pdb.set_trace()
                sys.exit()
            else:    
                for l in range(0,no_layers+1):
                    ind = 'l_' + str(l+1)
                    verified_spikes = Spikes_list[ind]
                    temp_list = Neural_Spikes[str(l)]
                    temp_list = temp_list[0]
                    correct_spikes = temp_list
                    correct_spikes = correct_spikes[:,cascade_count]
                    norm_spike_layers[l,itr_T] = norm_spike_layers[l,itr_T] + sum(abs(np.sign(verified_spikes)-np.sign(correct_spikes)))
                    norm_spike_layers_det[l,itr_T] = norm_spike_layers_det[l,itr_T] + sum(abs(np.sign(verified_spikes)-np.sign(correct_spikes)))
                    #pdb.set_trace()
        itr_T = itr_T + 1
        
        #.................Display and Write the Results to a File..................                   
    
    for l in range(0,no_layers+1):
        file_name = file_name_base_results + "/VerifiedSpikes/Spike_acc_vs_T_l_" + str(l) + "_%s.txt" %file_name_ending2
        temp = np.vstack([T_range,norm_spike_layers_det[l,:]])
        np.savetxt(file_name,temp.T,'%5.3f',delimiter='\t')
        #----------------------------------------------------------------------
#==============================================================================
norm_spike_layers_det = norm_spike_layers_det/float(ensemble_size-ensemble_count_init)

    
file_name_ending_temp = file_name_ending + '_I_' + str(inference_method)
file_name_ending2 = file_name_ending_temp + "_%s" %str(adj_fact_exc)
file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)


for l in range(0,no_layers+1):
    
    file_name = file_name_base_results + "/VerifiedSpikes/Avg_Spike_acc_vs_T_l_" + str(l) + "%s.txt" %file_name_ending2
    temp = np.vstack([T_range,np.divide(norm_spike_layers_det[l,:],T_range)])
    np.savetxt(file_name,temp.T,'%5.3f',delimiter='\t')
    
    file_name_ending23 = file_name_ending2 +"_%s" %str(T)
    file_name = file_name_base_results + "/Plot_Results/Per_Layer_Spike_acc_MLFF_%s.txt" %file_name_ending23
    running_time_file = open(file_name,'a')
    running_time_file.write("%d \t %f \n" %(l,norm_spike_layers[l,itr_T-1]))
    running_time_file.close()
