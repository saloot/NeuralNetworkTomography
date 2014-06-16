#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 320
n_inh_default = 80
connection_prob_default = 0.15
frac_input_neurons_default = 0.23
no_cascades_default = 1501
ensemble_size_default = 1
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

os.chdir('/Hesam/Academic/Network Tomography/NeuralAssociativeMemory/Neural_Network_Tomography/Python Codes/')
from auxiliary_functions import determine_binary_threshold
from auxiliary_functions import q_func_scalar
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:")
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

synapse_delay = 0.0                     # The synaptic delay of ALL links in ms
theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_minus = connection_prob * (float(n_inh)/float(n))
p_plus = connection_prob * (float(n_exc)/float(n))

file_name_base_data = "./Data"       #The folder to read neural data from
file_name_base_results = "./Results"       #The folder to store resutls

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
epsilon_plus_t = zeros(1+floor((no_cascades)/400))
epsilon_minus_t = zeros(1+floor((no_cascades)/400))

for ensemble_count in range(0,ensemble_size):

#======================READ THE NETWORK AND SPIKE TIMINGS======================
    
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "n_exc_%s" %str(int(n_exc))
    file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
    file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
    file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str(synapse_delay)
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #--------------------------------------------------------------------------
    
    
    #------------------------Read the Input Matrix-----------------------------
    file_name = file_name_base_data + "/Graphs/We_cascades_%s.txt" %file_name_ending
    We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    file_name = file_name_base_data + "/Graphs/Wi_cascades_%s.txt" %file_name_ending
    Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    W = np.vstack((We,Wi))
    #--------------------------------------------------------------------------
        

    #------------------------------Read and Sort Spikes------------------------
    file_name = file_name_base_data + "/Spikes/S_times_cascades_%s.txt" %file_name_ending
    
    S_time_file = open(file_name,'r')
    S_times = np.fromfile(file_name, dtype=float, sep='\t')
    S_time_file.close()
    
    neuron_count = 0
    cascade_count = 0
    in_spikes = np.zeros([n,no_cascades])
    out_spikes = np.zeros([n,no_cascades])
    for l in range(0,len(S_times)):
        if (S_times[l] == -1.0):
            neuron_count = neuron_count + 1            
        elif (S_times[l] == -2.0):
            neuron_count = 0
            cascade_count = cascade_count + 1
        else:
            if (S_times[l] > 0.00001):        
                if (S_times[l] < 0.00015):
                    in_spikes[neuron_count,cascade_count] = 1
                else:
                    out_spikes[neuron_count,cascade_count] = 1
    #--------------------------------------------------------------------------            
            
    #---------------------Check for Conflicts in Data--------------------------
    if (sum(sum(np.multiply(in_spikes,out_spikes))) > 0):
        print('Error! Do something!')
        sys.exit()
    if (sum(in_spikes==0.0001) or sum(in_spikes==0.0002)):
        print('Error! Do something! Non-integer in_spikes')
        sys.exit()
        
    if (sum(out_spikes==0.0001) or sum(out_spikes==0.0002)):
        print('Error! Do something! Non-integer out_spikes')
        sys.exit()
    #--------------------------------------------------------------------------
        
#==============================================================================

    #--------------------Calculate the PRobabilities of Belief Updates--------------------

    q = sum(in_spikes)/no_cascades/n
    no_input_ones = q  * n
    mu = (p_plus - p_minus) * no_input_ones
    var = sqrt((p_plus * (1 - p_plus) - p_minus * (1-p_minus) + 2 * p_plus * p_minus) * no_input_ones)
    mu_p = (p_plus - p_minus) * (no_input_ones - 1)
    var_p = sqrt((p_plus * (1 - p_plus) - p_minus * (1-p_minus) + 2 * p_plus * p_minus) * (no_input_ones - 1) )

    p_W_plus_1 = q_func_scalar((theta - mu_p - 1)/var_p)
    p_W_zero_1 = q_func_scalar((theta - mu_p)/var_p)
    p_W_minus_1 = q_func_scalar((theta - mu_p + 1)/var_p)
                
                
    p_W_plus_0 = 1 - p_W_plus_1
    p_W_zero_0 = 1 - p_W_zero_1
    p_W_minus_0 = 1 - p_W_minus_1
                                
    e1= p_W_plus_1 - p_W_plus_0
    e2 = p_W_minus_1 - p_W_minus_0
    epsilon_plus_base = e1 - (e1-e2)/4
    epsilon_minus_base = e2 + (e1-e2)/4
    # p_W_plus = p_W_plus/p_plus
    # p_W_minus = p_W_minus/p_minus
    # p_W_zero = p_W_zero/(1-p_plus-p_minus)
    #----------------------------------------------------------------------------------------


#============================INFER THE CONNECTIONS=============================
    out_spikes_neg = -pow(-1,out_spikes)
    L = in_spikes.sum(axis=0)
    
    print "acc_plus \t acc_minus \t acc_zero"
    print "-----------------------------------------------"
    itr = 0
    for T in range(100, no_cascades, 400):
        L_temp = L[0:T]
        Delta = 1.0/T
        out_spikes_neg_temp = out_spikes_neg[:,0:T]
        in_spikes_temp = in_spikes[:,0:T]
        out_spikes_temp = out_spikes[:,0:T]
        W_inferred = np.zeros([n,n])
        for i in range(0,n):
            for j in range(0,n):
                temp = np.multiply(in_spikes_temp[i,:],out_spikes_neg_temp[j,:])
                #W_inferred[i,j] = sum(np.divide(temp,L_temp))
                W_inferred[i,j] = sum(temp)*Delta
                
            #W_inferred[i,i] = 0
#==============================================================================


#=============================TRANSFORM TO BINARY==============================
        W_binary = np.zeros([n,n])
        if (binary_mode == 1):
            q = sum(in_spikes_temp)/T/n
            r = sum(out_spikes_temp)/T/n
            
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
            for i in range(0,n):
                w_temp = W_inferred[:,i]
                w_temp_s = np.sort(W_inferred[:,i])
                ind = np.argsort(w_temp)
                w_temp = zeros(n)
                w_temp[ind[0:int(round(p_minus*n))+1]] = -1
                epsilon_minus_t[itr] = epsilon_minus_t[itr] +  w_temp_s[int(round(p_minus*n))]
                w_temp[ind[len(ind)-int(round(p_plus*n))+1:len(ind)+1]] = 1
                epsilon_plus_t[itr] = epsilon_plus_t[itr] +  w_temp_s[len(ind)-int(round(p_plus*n))+1]
                W_binary[:,i] = w_temp
        
        itr = itr + 1
        #---Process to make sure that all outgoing links have the same type----

        #----------------------------------------------------------------------
        
#==============================================================================
        

#=============================CALCULATE ACCURACY===============================
        acc_plus = float(sum(multiply(W_binary>0,W>0)))/float(sum(W>0))
        acc_minus = float(sum(multiply(W_binary<0,W<0)))/float(sum(W<0))
        acc_zero = float(sum(multiply(W_binary==0,W==0)))/float(sum(W==0))
        
        prec_plus = float(sum(multiply(W_binary>0,W>0)))/float(sum(W_binary>0))
        prec_minus = float(sum(multiply(W_binary<0,W<0)))/float(sum(W_binary<0))
        prec_zero = float(sum(multiply(W_binary==0,W==0)))/float(sum(W_binary==0))
        
        file_name_ending_new = file_name_ending + "_%s" %str(T)
        file_name_ending_new = file_name_ending_new + "_%s" %str(binary_mode )
        
        file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % acc_plus)
        acc_file.write("%f \t" % acc_minus)
        acc_file.write("%f \t" % acc_zero)
        acc_file.write("\n")
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % prec_plus)
        acc_file.write("%f \t" % prec_minus)
        acc_file.write("%f \t" % prec_zero)
        acc_file.write("\n")
        acc_file.close()
        
        result_screen = str(acc_plus)
        result_screen = result_screen + "\t %s" %str(acc_minus)
        result_screen = result_screen + "\t %s" %str(acc_zero) 
        print result_screen
        
        result_screen = str(prec_plus)
        result_screen = result_screen + "\t %s" %str(prec_minus)
        result_screen = result_screen + "\t %s" %str(prec_zero) 
        print result_screen
        print('\n')

#==============================================================================


#==============CALCULATE AND STORE THE RUNNING TIME OF THE CODE================
epsilon_plus_t = epsilon_plus_t/ensemble_size/n
epsilon_minus_t = epsilon_minus_t/ensemble_size/n
t1 = time()                             # Capture the timer to calculate the running time per ensemble
print "Total simulation time was %f s" %(t1-t0)
file_name = file_name_base_results + "/RunningTimes/T_%s.txt" %file_name_ending
running_time_file = open(file_name,'a')
running_time_file.write("%f \t" %(t1-t0))
running_time_file.write("\n")
running_time_file.close()
#==============================================================================