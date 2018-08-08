#=======================IMPORT THE NECESSARY LIBRARIES=========================
import time
import string
import random
import numpy as np
import sys,getopt,os
import resource
try:
    import matplotlib.pyplot as plt
except:
    print 'Matplotlib can not be initiated! Pas de probleme!'
import pdb
from copy import deepcopy

from CommonFunctions.auxiliary_functions_inference import *
from CommonFunctions.Neurons_and_Networks import *
# reload(CommonFunctions.auxiliary_functions_inference)
#from sklearn import metrics
import multiprocessing

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    import random
    return ''.join(random.choice(chars) for _ in range(size))

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hN:Q:T:S:D:A:F:R:L:M:B:X:Y:C:V:J:U:Z:b:p:j:o:f:")

T,no_neurons,file_name_spikes,file_name_base_results,inference_method,sparse_thr0,beta,\
alpha0,max_itr_optimization,bin_size,no_processes,block_size,neuron_range,class_sample_freq,\
kernel_choice,no_hidden_neurons,no_structural_connections,file_name_ground_truth = parse_commands_inf_algo(input_opts)
#==============================================================================

#==================DO SANITY CHECK ON THE ENTERED PARAMETERS===================
if not no_neurons:
    print 'Sorry you should specify the number of observed neurons'
    sys.exit()

if not T:
    print 'Sorry you should specify the duration of recorded samples in miliseconds'
    sys.exit()
    
if (kernel_choice!= 'E') and (kernel_choice!='D'):
    print 'Unknown kernel!'
    sys.exit()
#==============================================================================

#================================INITIALIZATIONS===============================

#---------------------Initialize Simulation Variables--------------------------
d_window = 2                                    # The time window the algorithm considers to account for pre-synaptic spikes
rand_sample_flag = 1                            # If 1, the spikes are sampled randomly on intervals
#kernel_choice = 'E'
no_avg_itr = 10                                 # Number of times the inference algorithm is run with different parameters to find the connectivtiy

no_itr_over_dataset = max_itr_optimization
max_itr_optimization = no_itr_over_dataset*int(T/float(block_size))

num_process = min(no_processes,multiprocessing.cpu_count())
block_size = min(block_size,T)
print id_generator()
#------------------------------------------------------------------------------

#-------------------------Initialize Inference Parameters----------------------
if len(neuron_range)>1:
    neuron_range = range(neuron_range[0],neuron_range[1])

inferece_params = [inference_method,alpha0,sparse_thr0,sparse_thr0,max_itr_optimization,d_window,beta,bin_size,class_sample_freq,rand_sample_flag,kernel_choice]
#------------------------------------------------------------------------------

#---------------------Read The Actual Grapgh If Possible-----------------------
#file_name = '../Data/Graphs/Moritz_Actual_Connectivity.txt'
#file_name = '../Results/Inferred_Graphs/W_Pll_Moritz_I_7_S_5_T_75000_0.txt'
if file_name_ground_truth:
    W_gt = np.genfromtxt(file_name_ground_truth, dtype=None, delimiter='\t')
    W_gt = W_gt.T
elif no_structural_connections:
    print 'Sorry! for the structural information to work, you must specify the file name of gorund truth.'
    sys.exit()
#------------------------------------------------------------------------------

#==============================================================================


#===============================READ THE SPIKES================================
    
#----------------------------Read and Sort Spikes------------------------------
if not file_name_spikes:
    file_name_spikes = '../Data/Spikes/Moritz_Spike_Times.txt'
    file_name_spikes = '/scratch/salavati/NeuralNetworkTomography/Network Tomography Toolbox/Data/Spikes/Moritz_Spike_Times.txt'
    #file_name_spikes = '../Data/Spikes/HC3_ec013_198_processed.txt'
    #file_name_spikes = '/scratch/salavati/NeuralNetworkTomography/Network Tomography Toolbox/Data/Spikes/HC3_ec013_198_processed.txt'
    
ll = file_name_spikes.split('/')
ll = ll[-1]
file_name_prefix = ll.split('.txt')
file_name_prefix = file_name_prefix[0]
#------------------------------------------------------------------------------
        
#---------------------Preprocess the Spikes If Necessary-----------------------
file_name_spikes2 = file_name_spikes[:-4] + '_file.txt'
if not os.path.isfile(file_name_spikes2):            
    out_spikes = np.genfromtxt(file_name_spikes, dtype=float, delimiter='\t')            
    spike_file = open(file_name_spikes2,'w')
    fire_matx = [''] * (T+1)
    LL = out_spikes.shape[0]
    nn = -1
    for l in range(0,LL):
        last_nn = nn
        nn = int(out_spikes[l,0])
        tt = int(1000*out_spikes[l,1])
        if tt<=T:
            temp = fire_matx[tt]
            try:
                if str(nn) not in temp:
                    if len(temp):
                        temp = temp + ' ' + str(nn)
                    else:
                        temp = str(nn)
                else:
                    print 'What the ...?'
                    #pdb.set_trace()
                        
                if tt<=T:
                    fire_matx[tt] = temp
            except:
                pdb.set_trace()
    spike_file.write('\n'.join(fire_matx))
    spike_file.close()
    
    #-------------Calculate the number of firings-----------
    no_firings = np.zeros([nn+1])
    for ik in range(0,nn+1):
        no_firings[ik] = len(np.nonzero(out_spikes[:,0]==ik)[0])
        
    #-------------------------------------------------------
    #pdb.set_trace()
    file_name =  "../Data/Spikes/No_Firings_" + file_name_prefix + '.txt'
    np.savetxt(file_name,no_firings,'%f',delimiter='\t')  
#------------------------------------------------------------------------------

#==============================================================================


#======================GENERATE LIST OF HIDDEN NEURONS=========================
if no_hidden_neurons:
        hidden_neurons_temp2 = np.random.permutation(no_neurons)
        hidden_neurons_temp = hidden_neurons_temp2[0:no_hidden_neurons]
        hidden_neurons_temp = list(hidden_neurons_temp)
else:
    hidden_neurons_temp = []
#==============================================================================


#============================INFER THE CONNECTIONS=============================
W_infer = np.zeros([no_neurons+1-len(hidden_neurons_temp),len(neuron_range)])
    
itr_n = 0
for n_ind in neuron_range:
    
    #print 'memory so far %s' %str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    t_start = time.time()                           # starting time of the algorithm
    

    #---------------------------Adjust Hidden Neurons-------------------------
    hidden_neurons  = deepcopy(hidden_neurons_temp)
    if n_ind in hidden_neurons_temp:        
        hidden_neurons.remove(n_ind)
        hidden_neurons.append(hidden_neurons_temp2[no_hidden_neurons])
    #-------------------------------------------------------------------------

    #--------------TAKE CARE OF STRUCTURAL INFORMATION-----------------------
    if no_structural_connections:
        w_act = W_gt[:,n_ind]
        zero_connections = np.where(w_act == 0)[0]
        zero_connections = zero_connections[0:no_structural_connections]
        hidden_neurons = hidden_neurons + zero_connections
    #-------------------------------------------------------------------------
    
    for it in range(0,no_avg_itr):
        W_temp,used_ram_temp,cost = inference_constraints_hinge_parallel(file_name_spikes2,T,block_size,no_neurons,n_ind,num_process,inferece_params,hidden_neurons)
        if not it:
            W_inferred = W_temp
            used_ram = used_ram_temp
        else:
            W_inferred = W_inferred + W_temp
            used_ram = used_ram + used_ram_temp

    W_inferred = np.array(W_inferred)
    
    W_inferred = np.reshape(W_inferred,[no_neurons-len(hidden_neurons)+1,1])
    
    #.........................Save the Belief Matrices.........................
    file_name_ending = 'I_' + str(inference_method) + '_S_' + str(float(sparse_thr0)) + '_T_' + str(int(T))
    file_name_ending += '_C_' + str(int(num_process)) + '_B_' + str(int(block_size))
    file_name_ending += '_K_' + kernel_choice + '_H_' + str(class_sample_freq)
    
    if bin_size:
        file_name_ending += '_bS_' + str(bin_size)
    
    file_name_ending += '_ii_' + str(no_itr_over_dataset)
    
    file_name_ending += '_' + str(n_ind)

    if no_hidden_neurons:
        file_name_ending += '_F_' + str(int(no_hidden_neurons))

    if no_structural_connections:
        file_name_ending += '_f_' + str(int(no_structural_connections))

    if no_structural_connections or no_hidden_neurons:
        file_name_ending += id_generator()

        
    file_name =  file_name_base_results + "/Inferred_Graphs/W_Pll_%s_%s.txt" %(file_name_prefix,file_name_ending)
    tmp = W_inferred/float(no_avg_itr)
    
    #pdb.set_trace()
    tmp = tmp/(0.0001+np.linalg.norm(tmp))
    tmp = tmp/(0.0001+np.abs(tmp).max())
    
    W_infer[:,itr_n] = tmp.ravel()
    
    
    np.savetxt(file_name,tmp.T,'%2.6f',delimiter='\t')
    #..........................................................................
    
    print 'Inference successfully completed for T = %s ms. The results are saved on %s' %(str(T/1000.0),file_name)
    
    #....................Store Spent Time and Memory............................
    t_end = time.time()                           # The ending time of the algorithm    
    file_name =  file_name_base_results + "/Spent_Resources/CPU_RAM_%s_%s.txt" %(file_name_prefix,file_name_ending)
    tmp = [T,(t_end-t_start)/float(no_avg_itr),used_ram/float(no_avg_itr)]
    np.savetxt(file_name,tmp,delimiter='\t')
    #..........................................................................
    
    #.......................Store Optimization Cost............................
    file_name =  file_name_base_results + "/Spent_Resources/Opt_Cost_%s_%s.txt" %(file_name_prefix,file_name_ending)
    tmp = np.reshape(cost,[1,len(cost)])
    np.savetxt(file_name,tmp,delimiter='\t')
    #..........................................................................
    
    #..........................Store Hidden Neurons.............................
    if no_hidden_neurons or no_structural_connections:
        file_name =  file_name_base_results + "/Inferred_Graphs/Hidden_or_Structured_Neurons_%s_%s.txt" %(file_name_prefix,file_name_ending)
        np.savetxt(file_name,hidden_neurons,delimiter='\t')
    #..........................................................................
    
    itr_n = itr_n + 1
    
#==============================================================================


file_name =  file_name_base_results + "/Inferred_Graphs/W_Pll_%s_%s_n_%s_%s.txt" %(file_name_prefix,file_name_ending,str(neuron_range[0]),str(neuron_range[-1]))
np.savetxt(file_name,W_infer,'%2.6f',delimiter='\t')  

    