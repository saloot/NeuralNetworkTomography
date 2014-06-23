#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.2
frac_input_neurons_default = 0.4
no_cascades_default = 8000
ensemble_size_default = 10
delay_max_default = 1.0
binary_mode_default = 2
file_name_base_result_default = "./Results/FeedForward"
inference_method_default = 0
ensemble_count_init_default = 0
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
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
from time import time
import numpy as np
import sys,getopt,os
import matplotlib.pyplot as plt

os.chdir('C:\Python27')
#os.chdir('/home/salavati/Desktop/Neural_Tomography')
from auxiliary_functions import determine_binary_threshold
from auxiliary_functions import q_func_scalar
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
            ensemble_count_init = int(arg)                           # The index of ensemble we start with
        elif opt == '-M':
            inference_method = int(arg)                           # The index of ensemble we start with 
        elif opt == '-h':
            print(help_message)
            sys.exit()
else:
    print('Code will be executed using default values')
 
#==============================================================================


#================================INITIALIZATIONS===============================

#------------Set the Default Values if Variables are not Defines---------------
if 'n_exc' not in locals():
    n_exc = n_exc_default
    print('ATTENTION: The default value of %s for n_exc is considered.\n' %str(n_exc))

if 'n_inh' not in locals():
    n_inh = n_inh_default
    print('ATTENTION: The default value of %s for n_inh is considered.\n' %str(n_inh))

if 'connection_prob' not in locals():    
    connection_prob = connection_prob_default
    print('ATTENTION: The default value of %s for connection_prob is considered.\n' %str(connection_prob))

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

if 'delay_max' not in locals():
    delay_max = delay_max_default;
    print('ATTENTION: The default value of %s for delay_max is considered.\n' %str(delay_max))

if 'ensemble_count_init' not in locals():
    ensemble_count_init = ensemble_count_init_default;
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))

if 'inference_method' not in locals():
    inference_method = inference_method_default;
    print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(inference_method))  
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
n = n_exc + n_inh                       # Total number of neurons in the output layer

in_spikes = np.zeros([n,no_cascades])
out_spikes = np.zeros([no_cascades])
W_inferred_our = np.zeros([n])                                                  # Initialize the inferred matrix (the matrix of belifs) using our algorithm
W_inferred_hebian = np.zeros([n])                                               # Initialize the final binary matrix using our algorithm
W_binary_our = np.zeros([n])                                                    # Initialize the final binary matrix using simple hebbian correlation algorithm
W_binary_hebian = np.zeros([n])                                                 # Initialize the final binary matrix using a different "binarification" method
W_binary_modified2 = np.zeros([n])                                              # Initialize the final binary matrix using another "binarification" method
#------------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_minus = connection_prob * (float(n_inh)/float(n))
p_plus = connection_prob * (float(n_exc)/float(n))

file_name_base_data = "./Data/FeedForward"              # The folder to read neural data from
file_name_base_results = "./Results/Feedforward"        # The folder to store resutls
delay_max = float(delay_max)                            # This is done to avoid any incompatibilities in reading the data files

T_range = range(50, no_cascades, 250)                  # The range of sample sizes considered to investigate the effect of sample size on the performance
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

for ensemble_count in range(0,ensemble_size):

    t0_ensemble = time()
    
    #--------------------------------------------------------------------------                
    #--------------------READ THE NETWORK AND SPIKE TIMINGS--------------------

    #.......................Construct Prpoper File Names.......................
    file_name_ending = "n_exc_%s" %str(int(n_exc))
    file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
    file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
    #file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #..........................................................................
    
    #...........................Read the Input Matrix..........................
    file_name = file_name_base_data + "/Graphs/We_FF_n_1_%s.txt" %file_name_ending
    We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    file_name = file_name_base_data + "/Graphs/Wi_FF_n_1_%s.txt" %file_name_ending
    Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    W = np.hstack((We,Wi))
    #..........................................................................    
    
    #...........................Read and Sort Spikes...........................
    file_name = file_name_base_data + "/Spikes/S_times_FF_n_1_cascades_%s.txt" %file_name_ending
    
    S_time_file = open(file_name,'r')
    S_times = np.genfromtxt(file_name, dtype=float, delimiter='\t')
    S_time_file.close()
    in_spikes.fill(0)
    out_spikes.fill(0)
    s = S_times.shape
    cascade_count = 0
    for l in range(0,s[0]):
        neuron_count = int(S_times[l,0])        
        if (neuron_count == -2.0):            
            cascade_count = cascade_count + 1
        else:
            in_spikes[neuron_count,cascade_count] = 1
    
    print(sum(np.sign(S_times)))
    file_name = file_name_base_data + "/Spikes/S_times_FF_n_1_out_cascades_%s.txt" %file_name_ending
    S_times.fill(0)
    S_time_file = open(file_name,'r')
    S_times = np.genfromtxt(file_name, dtype=float, delimiter='\t')
    S_time_file.close()

    
    s = S_times.shape    
    for l in range(0,s[0]):
        cascade_count = int(S_times[l,0])
        tt = round(10000*S_times[l,1])-1
        out_spikes[cascade_count] = tt

    print(sum(np.sign(S_times)))
    S_times.fill(0)
    #..........................................................................
    
    #--------------------------------------------------------------------------            
    #--------------------------------------------------------------------------
    
#==============================================================================



#============================INFER THE CONNECTIONS=============================
    
    #--------------------------In-Loop Initializations--------------------------
    out_spikes_orig = out_spikes
    out_spikes_s = np.sign(out_spikes)
    out_spikes_neg = -pow(-1,out_spikes_s)
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
    
    
    for T in T_range:
        
        #------------------------In-Loop Initializations-----------------------
        L_temp = L[0:T]
        Delta = 1.0/float(T)
        out_spikes_neg_temp = out_spikes_neg[0:T]
        in_spikes_temp = in_spikes[:,0:T]
        out_spikes_temp = out_spikes_s[0:T]
        #----------------------------------------------------------------------
        
        #--------------------Construct the Belief Matrix-----------------------
        W_inferred_our = np.dot(in_spikes_temp,out_spikes_neg_temp.T)*Delta
        W_inferred_hebian = np.dot((-pow(-1,np.sign(in_spikes_temp))),out_spikes_neg_temp.T)*Delta
        #----------------------------------------------------------------------
        
        #------------------------Save the Belief Matrices---------------------------
        file_name_ending2 = file_name_ending +"_%s" %str(T)
        file_name = file_name_base_results + "/Inferred_Graphs/W_FF_n_to_1_Cascades_Delay_%s.txt" %file_name_ending2
        np.savetxt(file_name,W_inferred_our,'%5.3f',delimiter='\t')

        file_name = file_name_base_results + "/Inferred_Graphs/W_Hebb_FF_n_to_1_Cascades_Delay_%s.txt" %file_name_ending2
        np.savetxt(file_name,W_inferred_hebian,'%5.3f',delimiter='\t')
        #---------------------------------------------------------------------------
        
        #-----------CALCULATE AND STORE THE RUNNING TIME OF THE CODE------------        
        t1 = time()                             # Capture the timer to calculate the running time per ensemble
        #print "Total simulation time was %f s" %(t1-t0_ensemble+t_base)
        file_name = file_name_base_results + "/RunningTimes/T_FF_n_1_%s.txt" %file_name_ending2
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
        B_inh_mean[itr] = temp.mean()
        B_inh_std[itr] = temp.std()
        B_inh_max[itr] = temp.max()
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
               
        file_name = file_name_base_results + "/BeliefQuality/BQ_FF_Cascades_FF_n_1_%s.txt" %file_name_ending2
        running_time_file = open(file_name,'a')
        running_time_file.write("%f \t %f \t %f \n" %(B_exc_mean[itr],B_inh_mean[itr],B_void_mean[itr]))
        running_time_file.close()
        
        file_name = file_name_base_results + "/BeliefQuality/BQ_FF_Max_Min_Cascades_FF_n_1_%s.txt" %file_name_ending2
        running_time_file = open(file_name,'a')
        running_time_file.write("%f \t %f \t %f \t %f \n" %(B_exc_min[itr],B_void_max[itr],B_void_min[itr],B_inh_max[itr]))
        running_time_file.close()
        
        
        file_name = file_name_base_results + "/BeliefQuality/BQ_FF_Std_Cascades_FF_n_1_%s.txt" %file_name_ending2
        running_time_file = open(file_name,'a')
        running_time_file.write("%f \t %f \t %f \n" %(B_exc_std[itr],B_inh_std[itr],B_void_std[itr]))
        running_time_file.close()
        
        file_name = file_name_base_results + "/Plot_Results/B_exc_mean_FF_Cascades_FF_n_1_%s.txt" %file_name_ending
        running_time_file = open(file_name,'a')
        running_time_file.write("%d \t %f \t %f \n" %(T,B_exc_mean[itr],B_exc_std[itr]))
        running_time_file.close()
    
        file_name = file_name_base_results + "/Plot_Results/B_inh_mean_FF_Cascades_FF_n_1_%s.txt" %file_name_ending
        running_time_file = open(file_name,'a')
        running_time_file.write("%d \t %f \t %f \n" %(T,B_inh_mean[itr],B_inh_std[itr]))
        running_time_file.close()
    
        file_name = file_name_base_results + "/Plot_Results/B_void_mean_FF_Cascades_FF_n_1_%s.txt" %file_name_ending
        running_time_file = open(file_name,'a')
        running_time_file.write("%d \t %f \t %f \n" %(T,B_void_mean[itr],B_void_std[itr]))
        running_time_file.close()
        #..........................................................................
    
        itr = itr + 1
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    print "B_exc_mean: %f    B_void_mean: %f    B_inh_mean: %f" %(B_exc_mean[itr-1],B_void_mean[itr-1],B_inh_mean[itr-1])
    print "B_exc_std: %f    B_void_std: %f    B_inh_std: %f\n" %(B_exc_std[itr-1],B_void_std[itr-1],B_inh_std[itr-1])
 
#==================================================================================
#==================================================================================

plt.plot(T_range,B_exc_min,'r')
plt.plot(T_range,B_void_max,'g')
plt.plot(T_range,B_void_min,'g--')
plt.plot(T_range,B_inh_max,'b')
plt.show()       

