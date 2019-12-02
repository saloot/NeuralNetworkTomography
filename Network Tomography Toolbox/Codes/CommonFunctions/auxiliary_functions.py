#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
#from brian import *
import numpy as np
import pdb
import random
import copy
import numpy.ma as ma
#==============================================================================

#==============================================================================
#================================Q_Func_Scalar=================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function implements a scalar Q-function, which is the integral of a
# Gaussian distribution
#------------------------------------------------------------------------------

def q_func_scalar(x):
    import math
    return 0.5 - 0.5 * math.erf(x/math.sqrt(2))
#==============================================================================        
#==============================================================================


#==============================================================================
#================================generate_file_name=================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function generates the proper file anme ending to save our results
# to proper files.
#------------------------------------------------------------------------------

def generate_file_name(file_name_ending_base,inference_method,we_know_topology,pre_synaptic_method,generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,delay_known_flag):

    file_name_ending23 = file_name_ending_base + '_I_' + str(inference_method)
    file_name_ending23 = file_name_ending23 + '_Loc_' + we_know_topology
    file_name_ending23 = file_name_ending23 + '_Pre_' + pre_synaptic_method
    file_name_ending23 = file_name_ending23 + '_G_' + generate_data_mode
    file_name_ending23 = file_name_ending23 + '_X_' + str(infer_itr_max)
    file_name_ending23 = file_name_ending23 + '_Q_' + str(frac_stimulated_neurons)
    if (sparsity_flag):
        file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
    file_name_ending2 = file_name_ending23 +"_T_%s" %str(T)
    if delay_known_flag == 'Y':
        file_name_ending2 = file_name_ending2 +"_DD_Y"
    
    return file_name_ending2
            
#==============================================================================        
#==============================================================================
            
#==============================================================================
#=================================Multi_Picks==================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function determines if a vector has multiple local maximums
#------------------------------------------------------------------------------

def multi_picks(v):
    temp = abs(v)
    m = temp.max()
    
    i = temp.argmax()
    temp[i] = '-inf'
    multi_flag = 0 
    m2 = temp.max()
    
    if (abs(m2/float(m)) > 0.8):
        multi_flag = 1
    
    return multi_flag
    

#==============================================================================
#==============================soft_threshold==================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function truncates values that are smaller than a threshold to zero.
#------------------------------------------------------------------------------

def soft_threshold(W,thr):
    WW = np.multiply(W-thr,W>thr) + np.multiply(W+thr,W<-thr)
    return WW

def soft_threshold_double(W,thr_pos,thr_neg):
    WW = np.multiply(W-thr_pos,W>thr_pos) + np.multiply(W+thr_neg,W<-thr_neg)
    return WW
#==============================================================================
#==============================================================================


#==============================================================================
#===========================READ SPIKES FROM FILE==============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function reads spikes for a multi-layer neural network and returns the
# results in the form of a dictionary for each layer.
#------------------------------------------------------------------------------
def read_spikes(file_name):
    Neural_Spikes = {}    
    
    firing_times_max = 0
    
    S_times = np.genfromtxt(file_name, dtype=float, delimiter='\t')
    s = S_times.shape
    
    for l in range(0,s[0]):        
        neuron_count = int(S_times[l,0])
                
        if (neuron_count >= 0):
            if str(neuron_count) in Neural_Spikes:
                Neural_Spikes[str(neuron_count)].append(S_times[l,1])
            else:
                Neural_Spikes[str(neuron_count)] = []
                Neural_Spikes[str(neuron_count)].append(S_times[l,1])
            
            if S_times[l,1] > firing_times_max:
                firing_times_max = S_times[l,1] 


    print(s[0])
            
    
    print(firing_times_max)
    return Neural_Spikes,firing_times_max
#==============================================================================
#==============================================================================


#==============================================================================
#==============COMBINE WEIGHT MATRICES FOR DIFFERENT LAYERS====================
#==============================================================================

#-------------------------------Descriptions-----------------------------------
# This function combines the weights matrices from different layers to create
# a big matrix for all the layers as a single adjacency matrix.

# INPUT:
#    Network: the object containing the information about the structure of neural graph (ground truth)

# OUTPUT:
#    W_tot:  the matrix containing the connections weights of the 'whole' neural graph (all the layers)
#    DD_tot: the matrix containing the connections delays of the 'whole' neural graph (all the layers)
#------------------------------------------------------------------------------

def combine_weight_matrix(Network):
    W_tot = []
    DD_tot = []
    
    for l_in in range(0,Network.no_layers):
        n_exc = Network.n_exc_array[l_in]
        n_inh = Network.n_inh_array[l_in]
        n = n_exc + n_inh
                
        W_temp = []
        D_temp = []
        for l_out in range(0,Network.no_layers):
            n_exc = Network.n_exc_array[l_out]
            n_inh = Network.n_inh_array[l_out]
            m = n_exc + n_inh
                        
            if (l_out < l_in):
                if len(W_temp):
                    W_temp = np.hstack([W_temp,np.zeros([n,m])])
                else:
                    W_temp = np.zeros([n,m])
                            
                if len(D_temp):
                    D_temp = np.hstack([D_temp,np.zeros([n,m])])
                else:
                    D_temp = np.zeros([n,m])
                            
            else:
                ind = str(l_in) + str(l_out);
                temp_list = Network.Neural_Connections[ind];
                W = temp_list[0]
                DD = temp_list[1]
                        
                if len(W_temp):
                    W_temp = np.hstack([W_temp,W])
                else:
                    W_temp = W
                        
                if len(D_temp):
                    D_temp = np.hstack([D_temp,DD])
                else:
                    D_temp = DD
                
            
        if len(W_tot):
            W_tot = np.vstack([W_tot,W_temp])
        else:
            W_tot = W_temp
                
        if len(DD_tot):
            DD_tot = np.vstack([DD_tot,D_temp])
        else:
            DD_tot = D_temp
        
    return W_tot,DD_tot
#==============================================================================
#==============================================================================

    
#==============================================================================
#================COMBINE SPIKE TIMES FOR DIFFERENT LAYERS======================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function combines the spike times for the neurons in different layers in
# a large matrix that contains spikes for all the neurons in the graph. 

# INPUT:
#    Network: the object containing the information about the structure of neural graph (ground truth)
#    T: the duration of recorded samples (in miliseconds)
#    Actual_Neural_Times: A dictionary containing the spike times (in seconds) for each neuron (dictionary keys are the neuron indices)

# OUTPUT:
#    out_spikes_tot: a dictionary containing the spike times for all neurons in the graph
#    out_spikes_tot_mat: a matrix of size T*n, where a 1 in entry (t,i) means neuron i has fired at time t (in miliseconds)
#------------------------------------------------------------------------------

def combine_spikes_matrix(Neural_Spikes,T,jitter=0,del_prob=0):
            
    
    spikes = {}    
    n = len(Neural_Spikes)
    spikes_mat_nonzero = np.zeros([n,T])
    non_zero_neurons = []
    n_tot = 0
    
    i = 0
    for item in Neural_Spikes:
        ttimes = np.array(Neural_Spikes[item])
        tt = ttimes.shape[0]
        if jitter>0:
            ttimes = ttimes + jitter*(np.random.rand(tt)-0.5)
            ttimes = np.multiply(ttimes,(ttimes>0).astype(int))
            
        if del_prob:                
            inds = np.random.randint(del_prob,size=tt)
            inds = (inds > 0).astype(int)
            ttimes = np.multiply(ttimes,inds)
                
        ttimes = ttimes[ttimes<T/1000.0]
        spikes[item] = ttimes        
        
        sps_inds = (1000*ttimes).astype(int)
        spikes_mat_nonzero[i,sps_inds] = 1
        
        non_zero_neurons.append(int(item))
        if int(item) > n_tot:
            n_tot = int(item)
          
        i = i + 1
        
    spikes_mat = np.zeros([n_tot+1,T])
    
    for i in range(0,len(non_zero_neurons)):
        j = non_zero_neurons[i]
        spikes_mat[j,:] = spikes_mat_nonzero[i,:]
        
    spikes_mat_nonzero_s = np.zeros([n,T])
    itr = 0
    for i in range(0,n_tot):
        if i in non_zero_neurons:
            spikes_mat_nonzero_s[itr,:] = spikes_mat[i,:]
            itr = itr + 1
        else:
            if sum(spikes_mat[i,:]) > 0:
                print('Oops! There is something wrong here!')
    
    non_zero_neurons.sort()
    return spikes_mat,spikes_mat_nonzero_s,non_zero_neurons
#==============================================================================
#==============================================================================



#==============================================================================
#================COMBINE SPIKE TIMES FOR DIFFERENT LAYERS======================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function combines the spike times for the neurons in different layers in
# a large matrix that contains spikes for all the neurons in the graph. 

# INPUT:
#    Network: the object containing the information about the structure of neural graph (ground truth)
#    l_out: the index of output layer (1,2,3,...)
#    generate_data_mode: 'F' for stimulate-and-observe scenario
#                        'R' for the general case, where all neurons in the network can be trigerred due to external traffic
#    T: the duration of recorded samples (in miliseconds)
#    Actual_Neural_Times: A dictionary containing the spike times (in seconds) for each neuron (dictionary keys are the neuron indices)
#    Neural_Spikes: dictionary containing the spike times as an array for the case that we have: generate_data_mode = 'F'

# OUTPUT:
#    in_spikes: a matrix for pre-synaptic neural states, where a 1 in entry (i,t) means neuron i has fired at time t (in miliseconds)
#    out_spikes: a matrix for post-synaptic neural states, where a 1 in entry (i,t) means neuron i has fired at time t (in miliseconds)
#    non_stimul_inds: if we have (generate_data_mode =='F'), this dictionary specifies the neurons that have not been stimulated in each stimulation round
#------------------------------------------------------------------------------

def combine_spikes_matrix_FF(Network,l_out,generate_data_mode,T,Neural_Spikes):
    
    
    n_exc = Network.n_exc_array[l_out]
    n_inh = Network.n_inh_array[l_out]
    m = n_exc + n_inh                
    
    temp_list = Neural_Spikes[str(l_out)]
    out_spikes = (temp_list[2])
            
    if (generate_data_mode != 'R'):
        out_spikes = (out_spikes>0).astype(int)
        out_spikes = out_spikes[:,0:T]
    else:
        temp_sp = Neural_Spikes['act_'+str(l_out)]
        out_spikes_tot = np.zeros([m,T])
        for i in range(0,m):
            ttimes = np.array(temp_sp[str(i)])
            
            ttimes = ttimes[ttimes<T/1000.0]
                        
            sps_inds = (1000*ttimes).astype(int)
            out_spikes_tot[i,sps_inds] = 1
        
    
        out_spikes = out_spikes_tot

    #~~~~~~~~~~~~~~Concatenate Pre-synaptic Spikes~~~~~~~~~~~~~
    n_tot = 0
    n_so_far = 0
    in_spikes = []
    for l_in in range(0,l_out):
        if (generate_data_mode != 'R'):
            temp_list = Neural_Spikes[str(l_in)]
            temp_list = temp_list[2]
        else:
            temp_list = Neural_Spikes['act_'+str(l_in)]
            
        n_exc = Network.n_exc_array[l_in]
        n_inh = Network.n_inh_array[l_in]
        n = n_exc + n_inh
        n_tot = n_tot + n_exc + n_inh
        if len(in_spikes):
            in_spikes_temp = temp_list
            
            if (generate_data_mode == 'R'):
                out_spikes_tot_mat = np.zeros([n,T])
                for i in range(0,n):
                    ttimes = np.array(in_spikes_temp[str(i)])
                    ttimes = ttimes[ttimes<T/1000.0]                    
                    
                    sps_inds = (1000*ttimes).astype(int)
                    out_spikes_tot_mat[i,sps_inds] = 1
                    
                
                in_spikes_temp = out_spikes_tot_mat
                
            in_spikes = np.concatenate([in_spikes,in_spikes_temp])
            
        else:
            if l_in == 0:
                in_spikes = temp_list
                
                    
                if (generate_data_mode != 'R'):
                    in_spikes = in_spikes[:,0:T]
                else:
                    temp_sp = Neural_Spikes['act_'+str(l_in)]
                    in_spikes = np.zeros([n,T])
                    for i in range(0,n):
                        ttimes = np.array(temp_sp[str(i)])
                        ttimes = ttimes[ttimes<T/1000.0]                    
                        
                        sps_inds = (1000*ttimes).astype(int)
                        in_spikes[i,sps_inds] = 1

        if (generate_data_mode != 'R'):# or (inference_method != 4):
            in_spikes = (in_spikes>0).astype(int)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    return in_spikes, out_spikes
    
#==============================================================================
#==============================================================================



#==============================================================================
#=============================ADD NOISE TO SPIKES==============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function adds some noise to neural spikes. The noise is of two different
# natrues: 1) Missing spikes
#          2) Jitter
#------------------------------------------------------------------------------
def spike_binning(spikes_mat,bin_size):
    
    
    #~~~~~~~~~~~~~~~~Initilizations~~~~~~~~~~~~~~~
    n,T = spikes_mat.shape
    L = int(T/float(bin_size))
    out_spikes = np.zeros([n,L])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    for l in range(0,L):
        out_spikes[:,l] = np.sum(spikes_mat[:,l*bin_size:(l+1)*bin_size],axis = 1)
        #pdb.set_trace()
        
    return out_spikes
#==============================================================================
#==============================================================================

#==============================================================================
#===============Count Avg and Minimum Number of Incoming Spikes================
# This function counts the average and minimum of incoming spikes for all 
# incoming connections of all neurons in the network.
#==============================================================================
def count_incoming_spikes(file_name_ground_truth):

    #----------------------Load Ground Truth Connectivity----------------------
    W = np.genfromtxt(file_name_ground_truth, dtype=None, delimiter='\t')
    W = W.T
    n,m = W.shape
    #--------------------------------------------------------------------------

    #------------------------Load Ground Truth Spikes--------------------------
    file_name_spikes = '../Data/Spikes/LIF_Spike_Times_file.txt'
    #out_spikes = np.genfromtxt(file_name_spikes, dtype=float, delimiter='\t')
    neurons_spikes = []
    with open(file_name_spikes, 'r') as f:
        lines = f.readlines()
        for i in range(0,len(lines)):
            a = (lines[i][:-1]).split(' ')
            neurons_spikes += a
    #--------------------------------------------------------------------------

    #-----------------------Calculate Spike Counts-----------------------------
    neuron_range = range(0,22)
    no_spikes_neurons_min = np.zeros(len(neuron_range))
    no_spikes_neurons_avg = np.zeros(len(neuron_range))
    no_spikes_neurons_max = np.zeros(len(neuron_range))

    no_spikes_neurons_outgoing = np.zeros(len(neuron_range))

    no_spikes = np.zeros([n])
    for n_ind in range(0,n):
        #d = (np.where(neurons_spikes==n_ind))
        #no_spikes[n_ind] = sum(d[0])
        d = neurons_spikes.count(str(n_ind))
        no_spikes[n_ind] = d

    for n_ind in neuron_range:
        W_ss = W[:,n_ind]
        no_spikes_neurons_outgoing[n_ind] = no_spikes[n_ind]

        incoming_connections = np.where(W_ss != 0)[0]
        no_incoming_spikes = []
        for i in incoming_connections:
            no_incoming_spikes.append(no_spikes[i])

        no_spikes_neurons_min[n_ind] = min(no_incoming_spikes)
        no_spikes_neurons_max[n_ind] = max(no_incoming_spikes)
        no_spikes_neurons_avg[n_ind] = np.mean(no_incoming_spikes)
    #--------------------------------------------------------------------------

    #----------------------------Plot the Results------------------------------
    bar_width = 0.14#*x_axis_values.mean()
    
    #x_axis_values = x_axis_values/1000.0
    neuron_range = np.array(neuron_range)
    plt.bar(neuron_range,no_spikes_neurons_outgoing,bar_width,color='r',label='Outgoing');    
    plt.bar(neuron_range + bar_width,no_spikes_neurons_min,bar_width,color='b',label='Min. Incoming');
    plt.bar(neuron_range + 2*bar_width,no_spikes_neurons_avg,bar_width,color='y',label='Avg. Incoming');
    plt.bar(neuron_range + 3*bar_width,no_spikes_neurons_max,bar_width,color='g',label='Max. Incoming');

    plt.xlabel('Neuron Index', fontsize=16)
    plt.ylabel('No. Spikes', fontsize=16)
    plt.legend(loc='upper right')
    plt.show();
    #--------------------------------------------------------------------------

    #------------------------------Save the Results----------------------------
    np.save('../Data/Spikes/LIF_Firing_Rates.npy', no_spikes)
    np.save('../Data/Spikes/Summary_Firing_Rates.npy', [no_spikes_neurons_outgoing,no_spikes_neurons_min,no_spikes_neurons_max,no_spikes_neurons_avg])
    #--------------------------------------------------------------------------

#==============================================================================
#==============================================================================



#==============================================================================
#=================REMAP CONNECTIONS BASED ON REFERENCE VECTOR==================
# The following function gets a vector as input and an array containing index 
# of connection we are sure are equal to zero.
#==============================================================================
def remap_connections(input_vector,structural_connections,no_items):

    import numpy as np
    tmp = np.zeros([no_items])
    itr_iij = 0
    for iij in range(0,no_items):
        if iij not in structural_connections:
            tmp[iij] = input_vector[itr_iij]
            itr_iij += 1
        
    return tmp
#==============================================================================
#==============================================================================


#==============================================================================
#=================REMAP CONNECTIONS BASED ON REFERENCE VECTOR==================
# The following function gets a vector as input and an array containing index 
# of connection we are sure are equal to zero.
#==============================================================================
def enforce_structural_connections(input_vector,structural_connections):

    tmp = input_vector
    for iij in structural_connections:
        tmp[iij] = 0
        
    return tmp
#==============================================================================
#==============================================================================