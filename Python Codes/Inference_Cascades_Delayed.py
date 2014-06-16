#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 320
n_inh_default = 80
connection_prob_default = 0.15
frac_input_neurons_default = 0.2
no_cascades_default = 10000
ensemble_size_default = 5
binary_mode_default = 2
delay_max_default = 1.0
file_name_base_result_default = "./Results/Recurrent"
inference_method_default = 0
inference_method = 0
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
help_message = help_message + "-D xxx: To specify the maximum delay for the neural connections in milliseconds. Default value = %s.\n" %str(delay_max_default)
help_message = help_message + "-A xxx: To specify the folder that stores the results. Default value = %s. \n" %str(file_name_base_result_default)
help_message = help_message + "-F xxx: To specify the method use for inference, 0 for ours, 1 for Hopfield. Default value = %s. \n" %str(inference_method_default)
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================

#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
import time
import numpy as np
import sys,getopt,os
from time import time
from scipy import sparse
import pickle

#os.chdir('C:\Python27')
#os.chdir('/home/salavati/Desktop/Neural_Tomography')
from auxiliary_functions import determine_binary_threshold
from auxiliary_functions import q_func_scalar
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:A:D:F:")
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
            starting_index = int(arg)                           # The index of ensemble we start with 
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
    
if 'file_name_base_results' not in locals():
    file_name_base_results = file_name_base_result_default;
    print('ATTENTION: The default value of %s for file_name_base_results is considered.\n' %str(file_name_base_results))

if 'delay_max' not in locals():
    delay_max = delay_max_default;
    print('ATTENTION: The default value of %s for delay_max is considered.\n' %str(delay_max))

if 'starting_index' not in locals():
    starting_index = 0;
    print('ATTENTION: The default value of %s for starting_index is considered.\n' %str(starting_index))  
#------------------------------------------------------------------------------


#--------------------------Initialize Other Variables--------------------------
n = n_exc + n_inh                       # Total number of neurons in the output layer


theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_minus = connection_prob * (float(n_inh)/float(n))
p_plus = connection_prob * (float(n_exc)/float(n))

file_name_base_data = "./Data/Recurrent"       #The folder to read neural data from
#file_name_base_results = "./Results/Recurrent"       #The folder to store resutls

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
T_range = range(100, no_cascades, 500)
epsilon_plus_t = np.zeros(len(T_range))
epsilon_minus_t = np.zeros(len(T_range))

sim_window = round(12*delay_max)

H = np.zeros([n,sim_window])
#compressed_spikes = np.zeros([n,no_cascades])
if delay_max>0:
    recorded_spikes = np.zeros([n,no_cascades*sim_window])
else:
    recorded_spikes = np.zeros([n,no_cascades])
    
cumulative_recorded_spikes = np.zeros([n,no_cascades*sim_window])

W_inferred_our = np.zeros([n,n])
W_binary_our = np.zeros([n,n])
W_inferred_hebian = np.zeros([n,n])
W_binary_hebian = np.zeros([n,n])
W_binary_modified = np.zeros([n,n])
W_binary_modified2 = np.zeros([n,n])

t_base = time()
t_base = t_base-t0

for ensemble_count in range(starting_index,ensemble_size):

    t0_ensemble = time()
    
#======================READ THE NETWORK AND SPIKE TIMINGS======================    
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "n_exc_%s" %str(int(n_exc))
    file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
    file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
    file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str((delay_max))
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #--------------------------------------------------------------------------
    
    
    #------------------------Read the Input Matrix-----------------------------
    file_name = file_name_base_data + "/Graphs/We_Recurrent_Cascades_Delay_%s.txt" %file_name_ending
    We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    file_name = file_name_base_data + "/Graphs/Wi_Recurrent_Cascades_Delay_%s.txt" %file_name_ending
    Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    W = np.vstack((We,Wi))
    #--------------------------------------------------------------------------
        

    #------------------------------Read and Sort Spikes------------------------
    #file_name_prepared_spikes = file_name_base_data + "/Spikes/Processed_Recurrent_Casc_recorded_spikes_%s.dat" %file_name_ending
    #if os.path.isfile(file_name_prepared_spikes):
    #    print 'Reading processed timing data from file'

    #    file_name = file_name_prepared_spikes = file_name_base_data + "/Spikes/Processed_Recurrent_Casc_recorded_spikes_%s.dat" %file_name_ending
    #    f = open(file_name,'r')
    #    temp = pickle.load(f)
    #    recorded_spikes = temp.todense()
    #    f.close()

    #    #recorded_spikes = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #    #recorded_spikes = np.load(file_name)

    #    #fp = np.memmap(file_name,dtype='int',mode='r',shape=recorded_spikes.shape)
    #    #recorded_spikes[:] = fp[:]
    #    #del fp
        
    #    file_name = file_name_prepared_spikes = file_name_base_data + "/Spikes/Processed_Recurrent_Casc_cummulative_spikes_%s.dat" %file_name_ending
    #    f = open(file_name,'r')
    #    temp = pickle.load(f)
    #    cumulative_recorded_spikes = temp.todense()
    #    f.close()
        
    #    #cumulative_recorded_spikes = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #    #cumulative_recorded_spikes = np.load(file_name)

    #    #fp = np.memmap(file_name,dtype='int',mode='r',shape=cumulative_recorded_spikes.shape)
    #    #cumulative_recorded_spikes[:] = fp[:]
    #    #del fp
        
    #    #file_name = file_name_prepared_spikes = file_name_base_data + "/Spikes/Processed_times_Recurrent_Cascades_compressed_spikes_%s.dat" %file_name_ending
    #    #compressed_spikes = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #    #compressed_spikes = np.load(file_name)
    #    #fp = np.memmap(file_name,dtype='int',mode='r',shape=compressed_spikes.shape)
    #    #compressed_spikes[:] = fp[:]
    #    #del fp
    #else:
    #    print 'Processing Timing Data'
    if 1:
        file_name = file_name_base_data + "/Spikes/S_times_Recurrent_Cascades_Delay_%s.txt" %file_name_ending
    
        S_time_file = open(file_name,'r')
        S_times = np.fromfile(file_name, dtype=float, sep='\t')
        S_time_file.close()
    
        neuron_count = 0
        cascade_count = 0

        if delay_max:
            recorded_spikes.fill(0)
            for i in range(0,no_cascades):            
                recorded_spikes[:,i*sim_window+sim_window-1] = -1
        else:
            recorded_spikes.fill(0) 
        #compressed_spikes.fill(0)
        cumulative_recorded_spikes.fill(0)
    
        for l in range(0,len(S_times)):
            if (S_times[l] == -1.0):
                neuron_count = neuron_count + 1
            elif (S_times[l] == -2.0):
                #ss = sum(H,axis=1)
                #for ii in range(0,n):
                #    if sum(H[ii]):
                #        recorded_spikes[neuron_count,cascade_count] = 1            
                neuron_count = 0
                cascade_count = cascade_count + 1
                #H = np.zeros([n,round(10*delay_max)])
            
            else:
                if (S_times[l]>0):
                    tt = round(10000*S_times[l])-1
                    if (tt>0):
                        recorded_spikes[neuron_count,(cascade_count)*sim_window+sim_window-1] = 0
                        #compressed_spikes[neuron_count,cascade_count] = 1
                    recorded_spikes[neuron_count,(cascade_count)*sim_window+tt] = 1
                    cumulative_recorded_spikes[neuron_count,(cascade_count)*sim_window+tt+1:(cascade_count+1)*sim_window] = cumulative_recorded_spikes[neuron_count,(cascade_count)*sim_window+tt+1:(cascade_count+1)*sim_window] + np.multiply(np.ones([sim_window-tt-1]),range(1,int(sim_window-tt)))
            
        #----------------------------------------------------------------------            


        #-----------------Store the Results on a Proper File-------------------
        #file_name = file_name_prepared_spikes = file_name_base_data + "/Spikes/Processed_Recurrent_Casc_recorded_spikes_%s.dat" %file_name_ending
        #f = open(file_name,'w')
        #pickle.dump(sparse.csr_matrix(recorded_spikes),f)
        #f.close()
        #np.savetxt(file_name,recorded_spikes,'%d',delimiter='\t',newline='\n')
        #recorded_spikes.dump(file_name)
        #fp = np.memmap(file_name,dtype='int',mode='w+',shape=recorded_spikes.shape)
        #fp[:] = recorded_spikes[:]
        #del fp
        
        #file_name = file_name_prepared_spikes = file_name_base_data + "/Spikes/Processed_Recurrent_Casc_cummulative_spikes_%s.dat" %file_name_ending
        #f =open(file_name,'w')
        #pickle.dump(sparse.csr_matrix(cumulative_recorded_spikes),f)
        #f.close()
        #np.savetxt(file_name,cumulative_recorded_spikes,'%1.4f',delimiter='\t',newline='\n')
        #cumulative_recorded_spikes.dump(file_name)
        #fp = np.memmap(file_name,dtype='float64',mode='w+',shape=cumulative_recorded_spikes.shape)
        #fp[:] = cumulative_recorded_spikes[:]
        #del fp
        
        #file_name = file_name_prepared_spikes = file_name_base_data + "/Spikes/Processed_Recurrent_Casc_compressed_spikes_%s.dat" %file_name_ending
        #np.savetxt(file_name,compressed_spikes,'%d',delimiter='\t',newline='\n')
        #compressed_spikes.dump(file_name)
        #fp = np.memmap(file_name,dtype='int',mode='w+',shape=compressed_spikes.shape)
        #fp[:] = compressed_spikes[:]
        #del fp
        
        #----------------------------------------------------------------------
        
#==============================================================================



#============================INFER THE CONNECTIONS=============================
    #L = in_spikes.sum(axis=0)
    
    print "acc_plus \t acc_minus \t acc_zero"
    print "-----------------------------------------------"
    itr = 0
    for T in T_range:
        recorded_spikes_temp = recorded_spikes[:,0:T*sim_window]
        cumulative_spikes_temp = cumulative_recorded_spikes[:,0:T*sim_window]
        #L_temp = L[0:T]
        Delta = 1.0
        
        
        W_inferred_hebian.fill(0)
        W_inferred_our.fill(0)
        W_inferred_our = np.dot(cumulative_spikes_temp,recorded_spikes_temp.T)*Delta 
        W_inferred_hebian = np.dot((-pow(-1,np.sign(cumulative_spikes_temp))),cumulative_spikes_temp.T)*Delta

        file_name_ending2 = file_name_ending +"_%s" %str(T)
        file_name = file_name_base_results + "/Inferred_Graphs/W_Recurrent_Cascades_Delay_%s.txt" %file_name_ending2
        np.savetxt(file_name,W_inferred_our,'%5.3f',delimiter='\t')

        file_name = file_name_base_results + "/Inferred_Graphs/W_Hebb_Recurrent_Cascades_Delay_%s.txt" %file_name_ending2
        np.savetxt(file_name,W_inferred_hebian,'%5.3f',delimiter='\t')
        #for i in range(0,n):
        #    out_spikes = recorded_spikes_temp[i,:]            
        #    out_spikes_neg = out_spikes #-pow(-1,out_spikes)
        #    for j in range(0,n):
        #        in_spikes = cumulative_spikes_temp[j,:]                
        #        temp = np.multiply(in_spikes,out_spikes_neg)
        #        #W_inferred[i,j] = sum(np.divide(temp,L_temp))
        #        W_inferred[j,i] = sum(temp)*Delta
        #        
            #W_inferred[i,i] = 0
#==============================================================================


#=============================TRANSFORM TO BINARY==============================
        W_binary_our.fill(0)
        W_binary_hebian.fill(0)
        W_binary_modified.fill(0)
        W_binary_modified2.fill(0)
        

        #W_inferred_our = W_inferred_our.T
        #W_inferred_hebian = W_inferred_hebian.T
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
                w_temp = W_inferred_our[:,i]
                w_temp_s = np.sort(W_inferred_our[:,i])
                ind = np.argsort(w_temp)
                w_temp = np.zeros(n)
                w_temp[ind[0:int(round(p_minus*n))+1]] = -1
                epsilon_minus_t[itr] = epsilon_minus_t[itr] +  w_temp_s[int(round(p_minus*n))]
                w_temp[ind[len(ind)-int(round(p_plus*n))+1:len(ind)+1]] = 1
                epsilon_plus_t[itr] = epsilon_plus_t[itr] +  w_temp_s[len(ind)-int(round(p_plus*n))+1]
                W_binary_our[:,i] = w_temp

                w_temp = W_inferred_hebian[:,i]
                w_temp_s = np.sort(W_inferred_hebian[:,i])
                ind = np.argsort(w_temp)
                w_temp = np.zeros(n)
                w_temp[ind[0:int(round(p_minus*n))+1]] = -1
                w_temp[ind[len(ind)-int(round(p_plus*n))+1:len(ind)+1]] = 1    
                W_binary_hebian[:,i] = w_temp
        
        itr = itr + 1
        #---Process to make sure that all outgoing links have the same type----
        total_belief = sum(W_inferred_our.T)
        temp1 = np.sort(total_belief)
        ind = np.argsort(total_belief)
        
        for i in ind[len(ind)-int(round(p_plus*n))+1:len(ind)+1]:
            w_temp = W_inferred_our[i,:]
            w_temp_s = np.sort(W_inferred_our[i,:])
            ind2 = np.argsort(w_temp)
            w_temp = np.zeros(n) 
            w_temp[ind2[len(ind2)-int(round(connection_prob*n))+1:len(ind2)+1]] = 1
            W_binary_modified[i,:] = w_temp
            
        for i in ind[0:int(round(p_minus*n))+1]:
            w_temp = W_inferred_our[i,:]
            w_temp_s = np.sort(W_inferred_our[i,:])
            ind2 = np.argsort(w_temp)
            w_temp = np.zeros(n) 
            w_temp[ind2[0:int(round(connection_prob*n))+1]] = -1
            W_binary_modified[i,:] = w_temp
        #----------------------------------------------------------------------

        #---Process to make sure that all outgoing links have the same type----
        for i in range(0,n):
            w_temp = W_binary_our[i,:]
            d = sum(w_temp)
            w_temp_s = np.sort(W_inferred_our[i,:])
            ind = np.argsort(w_temp)
            w_temp = np.zeros(n)
            if (d>2):
                w_temp[ind[len(ind)-int(round(connection_prob*n))+1:len(ind)+1]] = 1
                W_binary_modified2[i,:] = w_temp
                
            elif (d< -1):
                w_temp[ind[0:int(round(connection_prob*n))+1]] = -1
                W_binary_modified2[i,:] = w_temp
        #----------------------------------------------------------------------
        file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_Our_Recurrent_Cascades_Delay_%s.txt" %file_name_ending2
        np.savetxt(file_name,W_binary_our,'%d',delimiter='\t')
        file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_Hebb_Recurrent_Cascades_Delay_%s.txt" %file_name_ending2
        np.savetxt(file_name,W_binary_hebian,'%d',delimiter='\t')
        
        
        #-----------CALCULATE AND STORE THE RUNNING TIME OF THE CODE------------        
        t1 = time()                             # Capture the timer to calculate the running time per ensemble
        print "Total simulation time was %f s" %(t1-t0_ensemble+t_base)
        file_name = file_name_base_results + "/RunningTimes/T_Cascades_Delayed_%s.txt" %file_name_ending2
        running_time_file = open(file_name,'a')
        running_time_file.write("%f \t" %(t1-t0_ensemble+t_base))
        running_time_file.write("\n")
        running_time_file.close()
        #----------------------------------------------------------------------
        
#==============================================================================
#==============================================================================
        

#=============================CALCULATE ACCURACY===============================        
        W_binary = W_binary_our
        
        acc_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W>np.zeros([n,n]))))
        acc_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W<np.zeros([n,n]))))
        acc_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W==np.zeros([n,n]))))
        
        prec_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W_binary>np.zeros([n,n]))))
        prec_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W_binary<np.zeros([n,n]))))
        prec_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W_binary==np.zeros([n,n]))))
        
        file_name_ending_new = file_name_ending + "_%s" %str(T)
        file_name_ending_new = file_name_ending_new + "_%s" %str(binary_mode )
        file_name_ending_new = file_name_ending_new + "_I_%s" %str(inference_method)
        
        file_name = file_name_base_results + "/Accuracies/Rec_Delayed_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % acc_plus)
        acc_file.write("%f \t" % acc_minus)
        acc_file.write("%f \t" % acc_zero)
        acc_file.write("\n")
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Prec_Delayed_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % prec_plus)
        acc_file.write("%f \t" % prec_minus)
        acc_file.write("%f \t" % prec_zero)
        acc_file.write("\n")
        acc_file.close()
        
        print '---------------'
        result_screen = str(acc_plus)
        result_screen = result_screen + "\t %s" %str(acc_minus)
        result_screen = result_screen + "\t %s" %str(acc_zero) 
        print result_screen
        
        result_screen = str(prec_plus)
        result_screen = result_screen + "\t %s" %str(prec_minus)
        result_screen = result_screen + "\t %s" %str(prec_zero) 
        print result_screen
        print('\n')


        W_binary = W_binary_hebian
        
        acc_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W>np.zeros([n,n]))))
        acc_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W<np.zeros([n,n]))))
        acc_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W==np.zeros([n,n]))))
        
        prec_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W_binary>np.zeros([n,n]))))
        prec_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W_binary<np.zeros([n,n]))))
        prec_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W_binary==np.zeros([n,n]))))
        
        file_name_ending_new = file_name_ending + "_%s" %str(T)
        file_name_ending_new = file_name_ending_new + "_%s" %str(binary_mode )
        file_name_ending_new = file_name_ending_new + "_I_%s" %str(inference_method)
        
        file_name = file_name_base_results + "/Accuracies/Hebian_Rec_Delayed_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % acc_plus)
        acc_file.write("%f \t" % acc_minus)
        acc_file.write("%f \t" % acc_zero)
        acc_file.write("\n")
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Hebian_Prec_Delayed_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % prec_plus)
        acc_file.write("%f \t" % prec_minus)
        acc_file.write("%f \t" % prec_zero)
        acc_file.write("\n")
        acc_file.close()

        print 'Hebian results'
        result_screen = str(acc_plus)
        result_screen = result_screen + "\t %s" %str(acc_minus)
        result_screen = result_screen + "\t %s" %str(acc_zero) 
        print result_screen
        
        result_screen = str(prec_plus)
        result_screen = result_screen + "\t %s" %str(prec_minus)
        result_screen = result_screen + "\t %s" %str(prec_zero) 
        print result_screen
        print('\n')


        W_binary = W_binary_modified
        
        acc_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W>np.zeros([n,n]))))
        acc_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W<np.zeros([n,n]))))
        acc_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W==np.zeros([n,n]))))
        
        prec_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W_binary>np.zeros([n,n]))))
        prec_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W_binary<np.zeros([n,n]))))
        prec_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W_binary==np.zeros([n,n]))))
        
        file_name_ending_new = file_name_ending + "_%s" %str(T)
        file_name_ending_new = file_name_ending_new + "_%s" %str(binary_mode )
        file_name_ending_new = file_name_ending_new + "_I_%s" %str(inference_method)
        
        file_name = file_name_base_results + "/Accuracies/Modified_Rec_Delayed_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % acc_plus)
        acc_file.write("%f \t" % acc_minus)
        acc_file.write("%f \t" % acc_zero)
        acc_file.write("\n")
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Modified_Prec_Delayed_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % prec_plus)
        acc_file.write("%f \t" % prec_minus)
        acc_file.write("%f \t" % prec_zero)
        acc_file.write("\n")
        acc_file.close()

        print 'Modified binary results'
        result_screen = str(acc_plus)
        result_screen = result_screen + "\t %s" %str(acc_minus)
        result_screen = result_screen + "\t %s" %str(acc_zero) 
        print result_screen
        
        result_screen = str(prec_plus)
        result_screen = result_screen + "\t %s" %str(prec_minus)
        result_screen = result_screen + "\t %s" %str(prec_zero) 
        print result_screen
        print('\n')

        W_binary = W_binary_modified2
        
        acc_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W>np.zeros([n,n]))))
        acc_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W<np.zeros([n,n]))))
        acc_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W==np.zeros([n,n]))))
        
        prec_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W_binary>np.zeros([n,n]))))
        prec_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W_binary<np.zeros([n,n]))))
        prec_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W_binary==np.zeros([n,n]))))
        
        file_name_ending_new = file_name_ending + "_%s" %str(T)
        file_name_ending_new = file_name_ending_new + "_%s" %str(binary_mode )
        file_name_ending_new = file_name_ending_new + "_I_%s" %str(inference_method)
        
        file_name = file_name_base_results + "/Accuracies/Modified2_Rec_Delayed_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % acc_plus)
        acc_file.write("%f \t" % acc_minus)
        acc_file.write("%f \t" % acc_zero)
        acc_file.write("\n")
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Modified2_Prec_Delayed_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % prec_plus)
        acc_file.write("%f \t" % prec_minus)
        acc_file.write("%f \t" % prec_zero)
        acc_file.write("\n")
        acc_file.close()
        

        print 'Modified 2 results'
        result_screen = str(acc_plus)
        result_screen = result_screen + "\t %s" %str(acc_minus)
        result_screen = result_screen + "\t %s" %str(acc_zero) 
        print result_screen
        
        result_screen = str(prec_plus)
        result_screen = result_screen + "\t %s" %str(prec_minus)
        result_screen = result_screen + "\t %s" %str(prec_zero) 
        print result_screen
        print('\n')

    #acc_file1.close()
    #acc_file2.close()
    #acc_file3.close()
    #acc_file4.close()
    #acc_file5.close()
    #acc_file6.close()
    #acc_file7.close()
    #acc_file8.close()
#==============================================================================



