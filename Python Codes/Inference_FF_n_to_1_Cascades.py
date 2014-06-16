#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.15
frac_input_neurons_default = 0.6
no_cascades_default = 10000
ensemble_size_default = 8
delay_max_default = 0.0
binary_mode_default = 2
file_name_base_result_default = "./Results/FeedForward"
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
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
import time
import numpy as np
import sys,getopt,os


#os.chdir('C:\Python27')
os.chdir('/home/salavati/Desktop/Neural_Tomography')
from auxiliary_functions import determine_binary_threshold
from auxiliary_functions import q_func_scalar
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time.time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:A:D:")
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

if 'file_name_base_result' not in locals():
    file_name_base_result = file_name_base_result_default;
    print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_result))

if 'delay_max' not in locals():
    delay_max = delay_max_default;
    print('ATTENTION: The default value of %s for delay_max is considered.\n' %str(delay_max))    
#------------------------------------------------------------------------------


#--------------------------Initialize Other Variables--------------------------
n = n_exc + n_inh                       # Total number of neurons in the output layer

theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_minus = connection_prob * (float(n_inh)/float(n))
p_plus = connection_prob * (float(n_exc)/float(n))

file_name_base_data = "./Data/FeedForward"       #The folder to read neural data from
file_name_base_results = "./Results/Feedforward"       #The folder to store resutls
delay_max = float(delay_max)
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

T_range = range(100, no_cascades, 100)
epsilon_plus_t = np.zeros(len(T_range))
epsilon_minus_t = np.zeros(len(T_range))

W_inferred_our = np.zeros([n])
W_inferred_hebian = np.zeros([n])
W_binary_our = np.zeros([n])
W_binary_hebian = np.zeros([n])
W_binary_modified2 = np.zeros([n])
for ensemble_count in range(0,ensemble_size):

#======================READ THE NETWORK AND SPIKE TIMINGS======================
    
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "n_exc_%s" %str(int(n_exc))
    file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
    file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
    #file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #--------------------------------------------------------------------------
    
    
    #------------------------Read the Input Matrix-----------------------------
    file_name = file_name_base_data + "/Graphs/We_FF_n_1_%s.txt" %file_name_ending
    We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    file_name = file_name_base_data + "/Graphs/Wi_FF_n_1_%s.txt" %file_name_ending
    Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    W = np.hstack((We,Wi))
    #--------------------------------------------------------------------------
        

    #------------------------------Read and Sort Spikes------------------------
    file_name = file_name_base_data + "/Spikes/S_times_FF_n_1_cascades_%s.txt" %file_name_ending
    
    
    S_time_file = open(file_name,'r')
    S_times = np.fromfile(file_name, dtype=float, sep='\t')
    S_time_file.close()
    
    neuron_count = 0
    cascade_count = 0
    in_spikes = np.zeros([n,no_cascades])
    out_spikes = np.zeros([no_cascades])
    
    for l in range(0,len(S_times)):
        if (S_times[l] == -1.0):
            neuron_count = neuron_count + 1            
        elif (S_times[l] == -2.0):
            neuron_count = 0
            cascade_count = cascade_count + 1
        else:
            if (S_times[l] > 0.00001):                        
                in_spikes[neuron_count,cascade_count] = 1
    
    
    file_name = file_name_base_data + "/Spikes/S_times_FF_n_1_out_cascades_%s.txt" %file_name_ending
    S_time_file = open(file_name,'r')
    S_times = np.fromfile(file_name, dtype=float, sep='\t')
    S_time_file.close()
    cascade_count = 0
    for l in range(0,len(S_times)):
        if (S_times[l] == -2.0):
            cascade_count = cascade_count + 1
        else:
            if (S_times[l] > 0.00001):        
                out_spikes[cascade_count] = S_times[l]
                    
    #--------------------------------------------------------------------------            
            
        
#==============================================================================



#============================INFER THE CONNECTIONS=============================
    out_spikes_orig = out_spikes
    out_spikes_s = np.sign(out_spikes)
    out_spikes_neg = -pow(-1,out_spikes_s)
    temp = delay_max*(out_spikes_s==0)/4
    out_spikes_s2 = out_spikes_s + temp
    out_spikes_s2 = (out_spikes_s2)
    #out_spikes_neg = tanh(np.multiply(out_spikes_neg,out_spikes_s2))
    L = in_spikes.sum(axis=0)
    
    print "acc_plus \t acc_minus \t acc_zero"
    print "-----------------------------------------------"
    itr = 0
    for T in T_range:
        L_temp = L[0:T]
        Delta = 1.0
        out_spikes_neg_temp = out_spikes_neg[0:T]
        in_spikes_temp = in_spikes[:,0:T]
        out_spikes_temp = out_spikes_s[0:T]
        # W_inferred.fill(0)
        # W_inferred_hebian.fill(0)
        W_inferred_our = np.dot(in_spikes_temp,out_spikes_neg_temp.T)*Delta
        W_inferred_hebian = np.dot((-pow(-1,np.sign(in_spikes_temp))),out_spikes_neg_temp.T)*Delta
        #time.sleep(1)
        #print(norm(W_inferred_our-W_inferred_hebian))
#==============================================================================


#=============================TRANSFORM TO BINARY==============================
        W_binary_our.fill(0)
        W_binary_hebian.fill(0)        
        if (binary_mode == 1):
            q = sum(in_spikes_temp)/T/n
            r = sum(out_spikes_temp)/T
            
            epsilon_plus,epsilon_minus = determine_binary_threshold(n,p_plus,p_minus,theta,T,Delta,q)
            
            
            
            #epsilon_plus_adjusted = epsilon_plus * q/frac_input_neurons
            #epsilon_minus_adjusted = epsilon_minus * q/frac_input_neurons
            
            epsilon_plus_adjusted = epsilon_plus  * (2*r-1)/(2*p_W_zero_1-1)
            epsilon_minus_adjusted = epsilon_minus * (2*r-1)/(2*p_W_zero_1-1)
            
            temp = (W_inferred > epsilon_plus_adjusted)
            temp = temp.astype(int)            
            W_binary = W_binary + temp
            
            temp = (W_inferred < epsilon_minus_adjusted)
            temp = temp.astype(int)
            W_binary = W_binary - temp
        elif (binary_mode == 2):

            ind = np.argsort(W_inferred_our)
            w_temp = np.zeros([n])
            w_temp[ind[0:int(round(p_minus*n))+1]] = -1
            #epsilon_minus_t[itr] = epsilon_minus_t[itr] +  w_temp_s[int(round(p_minus*n))]
            w_temp[ind[len(ind)-int(round(p_plus*n))+1:len(ind)+1]] = 1
            #epsilon_plus_t[itr] = epsilon_plus_t[itr] +  w_temp_s[len(ind)-int(round(p_plus*n))+1]
            W_binary_our = w_temp

            
        
            ind2 = np.argsort(W_inferred_hebian)
            w_temp2 = np.zeros([n])
            w_temp2[ind2[0:int(round(p_minus*n))+1]] = -1
            #epsilon_minus_t[itr] = epsilon_minus_t[itr] +  w_temp_s[int(round(p_minus*n))]
            w_temp2[ind2[len(ind2)-int(round(p_plus*n))+1:len(ind2)+1]] = 1
            #epsilon_plus_t[itr] = epsilon_plus_t[itr] +  w_temp_s[len(ind)-int(round(p_plus*n))+1]
            W_binary_hebian = w_temp2
            #print(norm(W_binary_hebian-W_binary_our))
            # print(norm(W_inferred_hebian-W_inferred_our))
        
        itr = itr + 1
        #---Process to make sure that all outgoing links have the same type----

        #----------------------------------------------------------------------
        
#==============================================================================
        

#=============================CALCULATE ACCURACY===============================
        W_binary = W_binary_our
        prec_plus = float(sum(np.multiply(W_binary>np.zeros([n]),W>np.zeros([n]))))/float(sum(W_binary>np.zeros([n])))
        prec_minus = float(sum(np.multiply(W_binary<np.zeros([n]),W<np.zeros([n]))))/float(sum(W_binary<np.zeros([n])))
        prec_zero = float(sum(np.multiply(W_binary==np.zeros([n]),W==np.zeros([n]))))/float(sum(W_binary==np.zeros([n])))
        
        reca_plus = float(sum(np.multiply(W_binary>np.zeros([n]),W>np.zeros([n]))))/float(sum(W>np.zeros([n])))
        reca_minus = float(sum(np.multiply(W_binary<np.zeros([n]),W<np.zeros([n]))))/float(sum(W<np.zeros([n])))
        reca_zero = float(sum(np.multiply(W_binary==np.zeros([n]),W==np.zeros([n]))))/float(sum(W==np.zeros([n])))
        
        file_name_ending_new = file_name_ending + "_%s" %str(T)
        file_name_ending_new = file_name_ending_new + "_%s" %str(binary_mode )
        
        file_name = file_name_base_results + "/Accuracies/Rec_FF_n_1_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % reca_plus)
        acc_file.write("%f \t" % reca_minus)
        acc_file.write("%f \t" % reca_zero)
        acc_file.write("\n")
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Prec_FF_n_1_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % prec_plus)
        acc_file.write("%f \t" % prec_minus)
        acc_file.write("%f \t" % prec_zero)
        acc_file.write("\n")
        acc_file.close()
        
        result_screen = str(reca_plus)
        result_screen = result_screen + "\t %s" %str(reca_minus)
        result_screen = result_screen + "\t %s" %str(reca_zero) 
        
        print result_screen
        
        result_screen = str(prec_plus)
        result_screen = result_screen + "\t %s" %str(prec_minus)
        result_screen = result_screen + "\t %s" %str(prec_zero) 
        print result_screen
        print ('\n')



        W_binary = W_binary_hebian
        prec_plus = float(sum(np.multiply(W_binary>np.zeros([n]),W>np.zeros([n]))))/float(sum(W_binary>np.zeros([n])))
        prec_minus = float(sum(np.multiply(W_binary<np.zeros([n]),W<np.zeros([n]))))/float(sum(W_binary<np.zeros([n])))
        prec_zero = float(sum(np.multiply(W_binary==np.zeros([n]),W==np.zeros([n]))))/float(sum(W_binary==np.zeros([n])))
        
        reca_plus = float(sum(np.multiply(W_binary>np.zeros([n]),W>np.zeros([n]))))/float(sum(W>np.zeros([n])))
        reca_minus = float(sum(np.multiply(W_binary<np.zeros([n]),W<np.zeros([n]))))/float(sum(W<np.zeros([n])))
        reca_zero = float(sum(np.multiply(W_binary==np.zeros([n]),W==np.zeros([n]))))/float(sum(W==np.zeros([n])))
        
        file_name_ending_new = file_name_ending + "_%s" %str(T)
        file_name_ending_new = file_name_ending_new + "_%s" %str(binary_mode )
        
        file_name = file_name_base_results + "/Accuracies/Hebian_Rec_FF_n_1_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % reca_plus)
        acc_file.write("%f \t" % reca_minus)
        acc_file.write("%f \t" % reca_zero)
        acc_file.write("\n")
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Hebian_Prec_FF_n_1_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % prec_plus)
        acc_file.write("%f \t" % prec_minus)
        acc_file.write("%f \t" % prec_zero)
        acc_file.write("\n")
        acc_file.close()
        
        print 'Hebian results'
        result_screen = str(reca_plus)
        result_screen = result_screen + "\t %s" %str(reca_minus)
        result_screen = result_screen + "\t %s" %str(reca_zero) 
        
        print result_screen
        
        result_screen = str(prec_plus)
        result_screen = result_screen + "\t %s" %str(prec_minus)
        result_screen = result_screen + "\t %s" %str(prec_zero) 
        print result_screen
        print ('\n')
#==============================================================================


#==============CALCULATE AND STORE THE RUNNING TIME OF THE CODE================
epsilon_plus_t = epsilon_plus_t/ensemble_size/n
epsilon_minus_t = epsilon_minus_t/ensemble_size/n
t1 = time.time()                             # Capture the timer to calculate the running time per ensemble
print "Total simulation time was %f s" %(t1-t0)
file_name = file_name_base_results + "/RunningTimes/T_FF_n_1_%s.txt" %file_name_ending
running_time_file = open(file_name,'a')
running_time_file.write("%f \t" %(t1-t0))
running_time_file.write("\n")
running_time_file.close()
#==============================================================================
