#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.2
frac_input_neurons_default = 0.4
no_cascades_default = 8000
ensemble_size_default = 10
binary_mode_default = 4
delay_max_default = 1.0
file_name_base_result_default = "./Results/Recurrent"
inference_method_default = 0
ensemble_count_init_default = 0
network_type_default = 'F'
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
help_message = help_message + "-F xxx: To specify the ensemble index to start simulation. Default value = %s. \n" %str(ensemble_count_init_default)
help_message = help_message + "-M xxx: To specify the method use for inference, 0 for ours, 1 for Hopfield. Default value = %s. \n" %str(inference_method_default)
help_message = help_message + "-N xxx: To specify the network type to simulate. Default value = %s. \n" %str(network_type_default)
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


#os.chdir('C:\Python27')
#os.chdir('/home/salavati/Desktop/Neural_Tomography')
from auxiliary_functions import beliefs_to_binary
from auxiliary_functions import calucate_accuracy
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:A:D:F:M:N:")
if (input_opts):
    for opt, arg in input_opts:
        if opt == '-E':
            n_exc = int(arg)                                    # The number of excitatory neurons in the output layer
        elif opt == '-I':
                n_inh = int(arg)                                # The number of inhibitory neurons in the output layer
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
            inference_method = int(arg)                         # The index of ensemble we start with
        elif opt == '-N':
            network_type = str(arg)                             # The type of network to simulate
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

if 'ensemble_count_init' not in locals():
    ensemble_count_init = ensemble_count_init_default;
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))

if 'inference_method' not in locals():
    inference_method = inference_method_default;
    print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(inference_method))
    
if 'network_type' not in locals():
    network_type = network_type_default;
    print('ATTENTION: The default value of %s for network_type is considered.\n' %str(network_type))
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
n = n_exc + n_inh                       # Total number of neurons in the output layer
sim_window = round(12*delay_max)                     # This is the number of iterations performed within each cascade

if delay_max>0:
    recorded_spikes = np.zeros([n,no_cascades*sim_window])                      # This matrix captures the time slot in which each neuron has fired in each step
else:
    recorded_spikes = np.zeros([n,no_cascades])
    
cumulative_recorded_spikes = np.zeros([n,no_cascades*sim_window])               # This matrix takes into account the effect of neural history 

W_inferred_our = np.zeros([n,n])                                                # Initialize the inferred matrix (the matrix of belifs) using our algorithm
W_binary_our = np.zeros([n,n])                                                  # Initialize the final binary matrix using our algorithm
W_inferred_hebbian = np.zeros([n,n])                                             # Initialize the inferred matrix (the matrix of belifs) using simple hebbian correlation algorithm
W_binary_hebian = np.zeros([n,n])                                               # Initialize the final binary matrix using simple hebbian correlation algorithm
W_binary_modified = np.zeros([n,n])                                             # Initialize the final binary matrix using a different "binarification" method
W_binary_modified2 = np.zeros([n,n])                                            # Initialize the final binary matrix using another "binarification" method
#------------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)
network_type = 'F'                     # The type of the simulated network

p_minus = connection_prob * (float(n_inh)/float(n))
p_plus = connection_prob * (float(n_exc)/float(n))


file_name_base_results = "./Results/Recurrent"       #The folder to store resutls

if (network_type == 'R'):
    file_name_base_results = "./Results/Recurrent"                  #The folder that stores the resutls
    file_name_base_data = "./Data/Recurrent"                        #The folder to read neural data from
    file_name_base_plot = "./Results/Results/Plot_Results"          #The folder to store resutls =
    name_base = 'Recurrent_'    
    
elif (network_type == 'F'):
    file_name_base_results = "./Results/FeedForward"                 #The folder that stores the resutls
    file_name_base_data = "./Data/FeedForward"                       #The folder to read neural data from
    file_name_base_plot = "./Results/FeedForward/Plot_Results"       #The folder to store resutls =
    name_base = 'FF_n_to_1_'
    
    
delay_max = float(delay_max)                         # This is done to avoid any incompatibilities in reading the data files   

T_range = range(50, no_cascades, 250)               # The range of sample sizes considered to investigate the effect of sample size on the performance
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

t_base = time()
t_base = t_base-t0
#==============================================================================



#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
first_flag = 1
for ensemble_count in range(ensemble_count_init,ensemble_size):

    t0_ensemble = time()
    
    #--------------------------------------------------------------------------                
    #----------------------------READ THE NETWORK------------------------------
    
    #.......................Construct Prpoper File Names.......................
    file_name_ending = "n_exc_%s" %str(int(n_exc))
    file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
    file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
    #file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str((delay_max))
    file_name_ending3 = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending3 = file_name_ending3 + "_%s" %str(ensemble_count)    
    #..........................................................................
        
    #..........................Read the Input Matrix...........................
    if (network_type == 'R'):
        file_name = file_name_base_data + "/Graphs/We_Recurrent_Cascades_Delay_%s.txt" %file_name_ending3
        We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
        file_name = file_name_base_data + "/Graphs/Wi_Recurrent_Cascades_Delay_%s.txt" %file_name_ending3
        Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    elif (network_type == 'F'):
        file_name = file_name_base_data + "/Graphs/We_FF_n_1_%s.txt" %file_name_ending3
        We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
        file_name = file_name_base_data + "/Graphs/Wi_FF_n_1_%s.txt" %file_name_ending3
        Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    W = np.hstack((We,Wi))
    #..........................................................................
    
    for T in T_range:

        file_name_ending2 = file_name_ending + "_%s" %str(ensemble_count)
        file_name_ending2 = file_name_ending2 +"_%s" %str(T)
        Delta = 1.0/float(T)
        
        #.........................Save the Belief Matrices.........................
        file_name = file_name_base_results + "/Inferred_Graphs/W_"
        file_name = file_name + name_base + "Cascades_Delay_%s.txt" %file_name_ending2
        W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')

        file_name = file_name_base_results + "/Inferred_Graphs/W_Hebb_"
        file_name = file_name + name_base + "Cascades_Delay_%s.txt" %file_name_ending2
        W_inferred_hebbian = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        #..........................................................................
        
        
        #..........................In-Loop Initializations.........................
        W_binary_our.fill(0)
        W_binary_hebian.fill(0)
        W_binary_modified.fill(0)
        W_binary_modified2.fill(0)
        #q = sum(in_spikes_temp)/T/n
        #r = sum(out_spikes_temp)/T/n
        adj_fact_exc = 1.0 # 0.125 # 0.25 # 0.375 # 0.5 # 0.625 # 0.75 # 0.875 # 1 # 1.125 # 1.25 # 1.375 # 1.5 # 1.625 # 1.75    # 0--> Prec     Inf--> Recall
        adj_fact_inh = 1.0 # 0.125 # 0.25 # 0.375 # 0.5 # 0.625 # 0.75 # 0.875 # 1 # 1.125 # 1.25 # 1.375 # 1.5 # 1.625 # 1.75
        params = [adj_fact_exc,adj_fact_inh]
        #..........................................................................
            
    
        #.................Calculate the Binary Matrix From Beliefs.................
        W_binary_our,centroids_our = beliefs_to_binary(binary_mode,W_inferred_our,n,p_plus,p_minus,theta,T,Delta,params,0)
        W_binary_hebian,centroids_hebb = beliefs_to_binary(binary_mode,W_inferred_hebbian,n,p_plus,p_minus,theta,T,Delta,params,0)
        if (network_type == 'R'):
            W_binary_modified = beliefs_to_binary(3,W_inferred_our,n,p_plus,p_minus,theta,T,Delta,params,0)
            W_binary_modified2 = beliefs_to_binary(2,W_inferred_our,n,p_plus,p_minus,theta,T,Delta,params,1)
        #..........................................................................
        
        #.......................Store the Binary Matrices..........................
        file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
        file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
        file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
        
        file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_Our_"
        file_name = file_name + name_base + "Cascades_Delay_%s.txt" %file_name_ending2
        np.savetxt(file_name,W_binary_our,'%d',delimiter='\t')
        
        file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_Hebb_"
        file_name = file_name + name_base + "Cascades_Delay_%s.txt" %file_name_ending2
        np.savetxt(file_name,W_binary_hebian,'%d',delimiter='\t')

        file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_Our_"
        file_name = file_name + name_base + "Cascades_Delay_%s.txt" %file_name_ending2
        ww = W_inferred_our.ravel()
        ww = np.vstack([ww,np.zeros([len(ww)])])
        np.savetxt(file_name,ww.T,'%f',delimiter='\t')

        file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_Hebb_"
        file_name = file_name + name_base + "Cascades_Delay_%s.txt" %file_name_ending2
        ww = W_inferred_hebbian.ravel()
        ww = np.vstack([ww,np.zeros([len(ww)])])
        np.savetxt(file_name,ww.T,'%f',delimiter='\t')

        if (binary_mode == 4):
            file_name = file_name_base_results + "/Inferred_Graphs/Our_centroids_"
            file_name = file_name + name_base + "Cascades_Delay_%s.txt" %file_name_ending2
            centroids_our = np.vstack([centroids_our,np.zeros([3])])
            np.savetxt(file_name,centroids_our,'%f',delimiter='\t')
            
            file_name = file_name_base_results + "/Inferred_Graphs/Hebb_centroids_"
            file_name = file_name + name_base + "Cascades_Delay_%s.txt" %file_name_ending2
            centroids_hebb = np.vstack([centroids_hebb,np.zeros([3])])
            np.savetxt(file_name,centroids_hebb,'%f',delimiter='\t')
        #..........................................................................
        
#==============================================================================
        

#=============================CALCULATE ACCURACY===============================        
        file_name_ending_new = file_name_ending2 
        #file_name_ending_new = file_name_ending_new + "_I_%s" %str(inference_method)
       
        #---------Calculate and Display Recall & Precision for Our Method----------    
        recal,precision = calucate_accuracy(W_binary_our,W)

        file_name = file_name_base_results + "/Accuracies/Rec_"
        file_name = file_name + name_base + "%s.txt" %file_name_ending_new

        if (first_flag):
            acc_file = open(file_name,'w')
        else:
            acc_file = open(file_name,'a')
        acc_file.write("%f \t %f \t %f \n" %(recal[0],recal[1],recal[2]))        
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Prec_"
        file_name = file_name + name_base + "%s.txt" %file_name_ending_new
        
        if (first_flag):
            acc_file = open(file_name,'w')
        else:
            acc_file = open(file_name,'a')
        acc_file.write("%f \t %f \t %f \n" %(precision[0],precision[1],precision[2]))        
        acc_file.close()
            
        print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
        print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
        print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
        print '\n'
        #--------------------------------------------------------------------------
        
        #-------Calculate and Display Recall & Precision for Hebbian Method--------
        recal,precision = calucate_accuracy(W_binary_hebian,W)
    
        file_name = file_name_base_results + "/Accuracies/Hebian_Rec_"
        file_name = file_name + name_base + "%s.txt" %file_name_ending_new
        
        if (first_flag):
            acc_file = open(file_name,'w')
        else:
            acc_file = open(file_name,'a')
        acc_file.write("%f \t %f \t %f \n" %(recal[0],recal[1],recal[2]))        
        acc_file.close()
        
        file_name = file_name_base_results + "/Accuracies/Hebian_Prec_"
        file_name = file_name + name_base + "%s.txt" %file_name_ending_new
        if (first_flag):
            acc_file = open(file_name,'w')
        else:
            acc_file = open(file_name,'a')
        acc_file.write("%f \t %f \t %f \n" %(precision[0],precision[1],precision[2]))        
        acc_file.close()
        
        #print '-----------Hebbian method performance in ensemble %d & T = %d----------' %(ensemble_count,T)
        #print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
        #print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
        #print '\n'
        #--------------------------------------------------------------------------
        
        if (network_type == 'R'):
            #---Calculate Recall & Precision for a Different Binarification Method-----
            recal,precision = calucate_accuracy(W_binary_modified,W)
    
            file_name = file_name_base_results + "/Accuracies/Modified_Rec_"
            file_name = file_name + name_base + "%s.txt" %file_name_ending_new           
            if (first_flag):
                acc_file = open(file_name,'w')
            else:
                acc_file = open(file_name,'a')
            acc_file.write("%f \t %f \t %f \n" %(recal[0],recal[1],recal[2]))        
            acc_file.close()
            
            file_name = file_name_base_results + "/Accuracies/Modified_Prec_"
            file_name = file_name + name_base + "%s.txt" %file_name_ending_new
            if (first_flag):
                acc_file = open(file_name,'w')
            else:
                acc_file = open(file_name,'a')
            acc_file.write("%f \t %f \t %f \n" %(precision[0],precision[1],precision[2]))        
            acc_file.close()
            
            print '-------Modified binary method performance in ensemble %d & T = %d------' %(ensemble_count,T)
            print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
            print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
            print '\n'
            #--------------------------------------------------------------------------
    
    
            #---Calculate Recall & Precision for a Different Binarification Method-----
            recal,precision = calucate_accuracy(W_binary_modified2,W)
    
            file_name = file_name_base_results + "/Accuracies/Modified2_Rec_"
            file_name = file_name + name_base + "%s.txt" %file_name_ending_new
            if (first_flag):
                acc_file = open(file_name,'w')
            else:
                acc_file = open(file_name,'a')
            acc_file.write("%f \t %f \t %f \n" %(recal[0],recal[1],recal[2]))        
            acc_file.close()
        
            file_name = file_name_base_results + "/Accuracies/Modified2_Prec_"
            file_name = file_name + name_base + "%s.txt" %file_name_ending_new
            if (first_flag):
                acc_file = open(file_name,'w')
            else:
                acc_file = open(file_name,'a')
            acc_file.write("%f \t %f \t %f \n" %(precision[0],precision[1],precision[2]))        
            acc_file.close()
        
            print '-------Modified binary 2 method performance in ensemble %d & T = %d-----' %(ensemble_count,T)
            print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
            print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])        
            #--------------------------------------------------------------------------
        first_flag = 1
        print '=========================================================================='
        print '\n'
        
#==============================================================================    

