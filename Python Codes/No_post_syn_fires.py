#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 320
n_inh_default = 80
connection_prob_default = 0.15
frac_input_neurons_default = 0.2
no_cascades_default = 10000
ensemble_size_default = 4
delay_max = 1.0
binary_mode_default = 2
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
help_message = help_message + "-B xxx: To specify the binarification algorithm. Default value = %s. \n" %str(binary_mode_default)
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import time
import numpy as np
import sys,getopt,os
from time import time

os.chdir('C:\Python27')
from auxiliary_functions import determine_binary_threshold
from auxiliary_functions import q_func_scalar
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:D:A:")
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
#------------------------------------------------------------------------------


#--------------------------Initialize Other Variables--------------------------
n = n_exc + n_inh                       # Total number of neurons in the output layer

theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_minus = connection_prob * (float(n_inh)/float(n))
p_plus = connection_prob * (float(n_exc)/float(n))

file_name_base_data = "./Data/Recurrent"       #The folder to read neural data from
file_name_base_results = "./Results/Recurrent"       #The folder to store resutls
file_name_base_plot = "./Results/Plot_Results"       #The folder to store resutls =
name_base = 'Delayed_'
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if NEcessary---------------
if not os.path.isdir(file_name_base_results):
    os.makedirs(file_name_base_results)
    temp = file_name_base_results + '/Accuracies'
    os.makedirs(temp)
    temp = file_name_base_results + '/RunningTimes'
    os.makedirs(temp)
#------------------------------------------------------------------------------

#==============================================================================

for ensemble_count in range(0,ensemble_size):

#======================READ THE NETWORK AND SPIKE TIMINGS======================
    
    q_range = [0.1,0.15,0.2,0.25,0.3]
    T = 8100
    for q in q_range:
        
        #----------------------Construct Prpoper File Names------------------------
        file_name_ending = "n_exc_%s" %str(int(n_exc))
        file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
        file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
        file_name_ending = file_name_ending + "_r_%s" %str(q)
        file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
        file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
        file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
        file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
        #--------------------------------------------------------------------------
    
        

        #------------------------------Read and Count Spikes-----------------------
        file_name = file_name_base_data + "/Spikes/S_times_Recurrent_Cascades_Delay_%s.txt" %file_name_ending
        
        S_time_file = open(file_name,'r')
        S_times = np.fromfile(file_name, dtype=float, sep='\t')
        S_time_file.close()
    
        spike_count = sum(S_times>0.0001)
        #--------------------------------------------------------------------------   
        
        #----------------------Construct Prpoper File Names------------------------
        file_name_ending = name_base + "n_exc_%s" %str(int(n_exc))
        file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
        file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
        file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
        file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
        file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
        file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
        file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
        file_name_ending = file_name_ending + "_%s" %str(T)
        file_name_ending = file_name_ending + "_%s" %str(binary_mode)
        #--------------------------------------------------------------------------
    
        #----------------------------Read the Precisions---------------------------
        file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %file_name_ending
        Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        s = Acc.shape
        if (len(s) > 1):
            temp = (sum(Acc,axis=0))/float(s[0])
        else:
            temp = Acc
        
        P_C_plus = temp[0]
        P_C_minus = temp[1]
        P_C_zero = temp[2]
        #--------------------------------------------------------------------------
        
        #-----------------------Write the Results on a File------------------------        
        spike_count = float(spike_count)/float(T)/float(n)
        file_name_ending = "n_exc_%s" %str(int(n_exc))
        file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
        file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
        #file_name_ending = file_name_ending + "_r_%s" %str(q)
        #file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
        file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
        #file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
        file_name_ending = file_name_ending + "_%s" %str(ensemble_count)        
        file_name_ending = file_name_ending + "_%s" %str(T)        
        
        file_name = file_name_base_data + "/Spikes/No_Spikes_vs_Reca_exc_%s.txt" %file_name_ending
        sp_count_file = open(file_name,'a')
        sp_count_file.write("%f \t" % spike_count)
        sp_count_file.write("%f \t" % P_C_plus)        
        sp_count_file.write("\n")
        sp_count_file.close()
        
        file_name = file_name_base_data + "/Spikes/No_Spikes_vs_Reca_inh_%s.txt" %file_name_ending
        sp_count_file = open(file_name,'a')
        sp_count_file.write("%f \t" % spike_count)
        sp_count_file.write("%f \t" % P_C_minus)        
        sp_count_file.write("\n")
        sp_count_file.close()
        
        file_name = file_name_base_data + "/Spikes/No_Spikes_vs_Reca_zero_%s.txt" %file_name_ending
        sp_count_file = open(file_name,'a')
        sp_count_file.write("%f \t" % spike_count)
        sp_count_file.write("%f \t" % P_C_zero)       
        sp_count_file.write("\n")
        sp_count_file.close()
        #--------------------------------------------------------------------------
        
#==============================================================================

