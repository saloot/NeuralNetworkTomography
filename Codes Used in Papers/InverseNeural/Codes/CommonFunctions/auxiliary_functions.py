#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
#from brian import *
import numpy as np
from scipy import sparse
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
#==============================================================================
#==============================================================================


#==============================================================================
#===========================READ SPIKES FROM FILE==============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function reads spikes for a multi-layer neural network and returns the
# results in the form of a dictionary for each layer.
#------------------------------------------------------------------------------
def read_spikes(file_name_base,no_layers,n_exc_array,n_inh_array,params):
    Neural_Spikes = {}
    store_method = params[0]
    
    if store_method == 'c':
        no_stimul_rounds = params[1]
        sim_window = params[2]
        
        for l_in in range(0,no_layers):        
            file_name = file_name_base + '_l_' + str(l_in) +'.txt'        
            S_times = np.genfromtxt(file_name, dtype=float, delimiter='\t')
            
            n_exc = n_exc_array[l_in]
            n_inh = n_inh_array[l_in]
            n = n_exc + n_inh
                
            in_spikes = np.zeros([n,no_stimul_rounds])
            in_spikes.fill(0)
    
            recorded_spikes = np.zeros([n,no_stimul_rounds*sim_window])                      # This matrix captures the time slot in which each neuron has fired in each step
            cumulative_recorded_spikes = np.zeros([n,no_stimul_rounds*sim_window])               # This matrix takes into account the effect of neural history 
            recorded_spikes.fill(0)
            for i in range(0,no_stimul_rounds):            
                recorded_spikes[:,i*sim_window+sim_window-1] = -1
    
            s = S_times.shape
            cascade_count = 0
            for l in range(0,s[0]):
                neuron_count = int(S_times[l,0])        
                if (neuron_count == -2.0):            
                    cascade_count = cascade_count + 1
                else:
                    #tt = mod(round(10000*S_times[l,1]),sim_window)-1
                    #in_spikes[neuron_count,round(10000*S_times[l,1])/sim_window] = tt;#S_times[l,1]
                    tt = round(1000*S_times[l,1])-1   #             tt = round(10000*S_times[l,1])-1                
                    in_spikes[neuron_count,cascade_count] = S_times[l,1]
                    if (tt>0):
                        recorded_spikes[neuron_count,(cascade_count)*sim_window+sim_window-1] = 0
                    
                    recorded_spikes[neuron_count,(cascade_count)*sim_window+tt] = 1
                    cumulative_recorded_spikes[neuron_count,(cascade_count)*sim_window+tt+1:(cascade_count+1)*sim_window] = cumulative_recorded_spikes[neuron_count,(cascade_count)*sim_window+tt+1:(cascade_count+1)*sim_window] + np.divide(np.ones([sim_window-tt-1]),range(1,int(sim_window-tt)))
                    #cumulative_recorded_spikes[neuron_count,(cascade_count)*sim_window+tt+1:(cascade_count+1)*sim_window] = np.ones([sim_window-tt-1])

            print(sum((S_times>0)))        
            Neural_Spikes[str(l_in)] = list([recorded_spikes,cumulative_recorded_spikes,in_spikes]) #in_spikes
    else:
        no_stimul_rounds = params[1]
        n_tot = sum(n_exc_array)+sum(n_inh_array)
        tot_spikes = np.zeros([n_tot,no_stimul_rounds])
        tot_spikes.fill(0)
        n_so_far = 0
        for l_in in range(0,no_layers):        
            file_name = file_name_base + '_l_' + str(l_in) +'.txt'        
            S_times = np.genfromtxt(file_name, dtype=float, delimiter='\t')
            
            n_exc = n_exc_array[l_in]
            n_inh = n_inh_array[l_in]
            n = n_exc + n_inh
                
            in_spikes = np.zeros([n,no_stimul_rounds])
            in_spikes.fill(0)
    
            s = S_times.shape
            
            firing_times = {}
            for ii in range(0,n):                
                firing_times[str(ii)] = []
            
            firing_times_max = 0    
            for l in range(0,s[0]):
                neuron_count = int(S_times[l,0])
                
                if (neuron_count >= 0):                    
                    tt = round(1000*S_times[l,1])-1               # tt = round(10000*S_times[l,1])-1                
                    in_spikes[neuron_count,tt] = 1
                    tot_spikes[neuron_count+n_so_far,tt] = 1
                    if S_times[l,1] > firing_times_max:
                        firing_times_max = S_times[l,1]
                    firing_times[str(neuron_count)].append(S_times[l,1])
                    
            n_so_far = n_so_far + n
            Neural_Spikes[str(l_in)] = list([[],[],in_spikes])
            Neural_Spikes['tot'] = tot_spikes
            print(sum(sum(in_spikes)))
            print firing_times_max
            Neural_Spikes['act_' + str(l_in)] = firing_times
    
    return Neural_Spikes,firing_times_max
#==============================================================================
#==============================================================================


#==============================================================================
#=========================BELIEF QUALITY ASSESSMENT============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function calculates the quality of the computed beliefs about the neural
# graph

# INPUT:
#    W_inferred: the inferred graph (association matrix)
#    W_orig: the actual graph (ground truth)

# OUTPUT:
#    mean_beliefs:    the avaerage beliefs for inhibitory, non-existent and excitatory connections
#    max_min_beliefs: the maximum and minimul value of beliefs for inhibitory, non-existent and excitatory connections
#------------------------------------------------------------------------------
def calculate_belief_quality(W_inferred,W_orig):

    #------------------------------Initialization------------------------------
    ss = W_inferred.shape
    m = ss[0]
    n = ss[1]
    
    tt = W_orig.shape
    
    if (ss!=tt):
        print('Error! The original and the ifnerred graphs should have the same size.')
        sys.exit()
        
    B_exc_mean = np.zeros([n,1])
    B_inh_mean = np.zeros([n,1])
    B_void_mean = np.zeros([n,1])
    
    B_exc_min = np.zeros([n,1])
    B_inh_max = np.zeros([n,1])
    B_void_max = np.zeros([n,1])
    B_void_min = np.zeros([n,1])
    #--------------------------------------------------------------------------
                
    if 1:
        
        for i in range(0,n):
        
            #--------Caclulate the Minimum Value of the Excitatory Beliefs---------
            a = np.nonzero(W_orig[:,i]>0)
            temp = W_inferred[a,i]
            if len(temp[0]):
                B_exc_mean[i] = temp.mean()        
                B_exc_min[i] = temp.min()
            else:
                B_exc_mean[i] = float('NaN')
                B_exc_min[i] = float('NaN') 
            #----------------------------------------------------------------------
    
            #--------Caclulate the Maximum Value of the Inhibitory Beliefs---------
            a = np.nonzero(W_orig[:,i]<0)
            temp = W_inferred[a,i]        
            if (len(temp[0]) > 0):
                B_inh_mean[i] = temp.mean()            
                B_inh_max[i] = temp.max()
            else:
                B_inh_mean[i] = float('NaN') #B_inh_mean[itr-1]            
                B_inh_max[i] = float('NaN') #B_inh_max[itr-1]
            #----------------------------------------------------------------------
    
            #------Caclulate the Minimum and Maximum Value of the Void Beliefs-----
            a = np.nonzero(W_orig[:,i]==0)
            temp = W_inferred[a,i]
            B_void_mean[i] = temp.mean()
            B_void_max[i] = temp.max()
            B_void_min[i] = temp.min()
            #----------------------------------------------------------------------
    else:
       
    
        #--------Caclulate the Minimum Value of the Excitatory Beliefs---------
        a = np.nonzero(W_orig>0)
        #pdb.set_trace()
        temp = W_inferred[a]
        #pdb.set_trace()
        if len(temp):
            B_exc_mean[0] = temp.mean()        
            B_exc_min[0] = temp.min()
        else:
            B_exc_mean[0] = float('NaN')
            B_exc_min[0] = float('NaN') 
        #----------------------------------------------------------------------
    
        #--------Caclulate the Maximum Value of the Inhibitory Beliefs---------
        a = np.nonzero(W_orig<0)
        temp = W_inferred[a]        
        if (len(temp) > 0):
            B_inh_mean[0] = temp.mean()            
            B_inh_max[0] = temp.max()
        else:
            B_inh_mean[0] = float('NaN') #B_inh_mean[itr-1]            
            B_inh_max[0] = float('NaN') #B_inh_max[itr-1]
        #----------------------------------------------------------------------
    
        #------Caclulate the Minimum and Maximum Value of the Void Beliefs-----
        a = np.nonzero(W_orig==0)
        temp = W_inferred[a]
        B_void_mean[0] = temp.mean()
        B_void_max[0] = temp.max()
        B_void_min[0] = temp.min()
        #----------------------------------------------------------------------

    
    mean_beliefs = np.hstack([B_exc_mean,B_void_mean,B_inh_mean])
    max_min_beliefs = np.hstack([B_exc_min,B_void_max,B_void_min,B_inh_max])
    
    return mean_beliefs,max_min_beliefs
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
#    generate_data_mode: 'F' for stimulate-and-observe scenario
#                        'R' for the general case, where all neurons in the network can be trigerred due to external traffic

# OUTPUT:
#    W_tot:  the matrix containing the connections weights of the 'whole' neural graph (all the layers)
#    DD_tot: the matrix containing the connections delays of the 'whole' neural graph (all the layers)
#------------------------------------------------------------------------------

def combine_weight_matrix(Network,generate_data_mode):
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
#    generate_data_mode: 'F' for stimulate-and-observe scenario
#                        'R' for the general case, where all neurons in the network can be trigerred due to external traffic
#    T: the duration of recorded samples (in miliseconds)
#    Actual_Neural_Times: A dictionary containing the spike times (in seconds) for each neuron (dictionary keys are the neuron indices)
#    Neural_Spikes: dictionary containing the spike times as an array for the case that we have: generate_data_mode = 'F'

# OUTPUT:
#    out_spikes_tot: a dictionary containing the spike times for all neurons in the graph
#    out_spikes_tot_mat: a matrix of size T*n, where a '1' in entry (t,i) means neuron i has fired at time t (in miliseconds)
#    non_stimul_inds: if we have (generate_data_mode =='F'), this dictionary specifies the neurons that have not been stimulated in each stimulation round
#------------------------------------------------------------------------------

def combine_spikes_matrix(Network,generate_data_mode,T,Actual_Neural_Times,Neural_Spikes):
        
    if generate_data_mode == 'R':                
        out_spikes_tot = {}
    else:        
        out_spikes_tot = []
            
    n_so_far = 0
    n_tot = 0
    for l_in in range(0,Network.no_layers):
        n_exc = Network.n_exc_array[l_in]
        n_inh = Network.n_inh_array[l_in]
        n = n_exc + n_inh
        n_tot = n_tot + n
    
    out_spikes_tot_mat = np.zeros([n_tot,T])
    
    for l_in in range(0,Network.no_layers):
        n_exc = Network.n_exc_array[l_in]
        n_inh = Network.n_inh_array[l_in]
        n = n_exc + n_inh
                
                
        if generate_data_mode == 'R':
            spikes_times = Actual_Neural_Times[str(l_in)]
            for i in range(0,n):
                ind = str(n_so_far)
                ttimes = np.array(spikes_times[str(i)])
                ttimes = ttimes[ttimes<T/1000.0]
                out_spikes_tot[ind] = ttimes
                    
                sps_inds = (1000*ttimes).astype(int)
                out_spikes_tot_mat[n_so_far,sps_inds] = 1
                        
                n_so_far = n_so_far + 1

        else:
            temp_list = Neural_Spikes[str(l_in)]
            spikes = (temp_list[2])
                
            if len(out_spikes_tot):
                out_spikes_tot = np.vstack([out_spikes_tot,spikes])
                
            else:
                out_spikes_tot = spikes
    
    
    non_stimul_inds = {}
    if generate_data_mode != 'R':
        
        n_layer_1 = Network.n_exc_array[0] + Network.n_inh_array[0]
        n_layer_2 = Network.n_exc_array[1] + Network.n_inh_array[1]
        out_spikes_tot = out_spikes_tot[:,0:T]
        
        for ttt in range(0,T):
            temp = out_spikes_tot[0:n_layer_1,ttt]
            ttemp = np.nonzero(temp<=0)
            ttemp = list(ttemp[0])
            ttemp.extend(range(n_layer_1,n_layer_1+n_layer_2))
            non_stimul_inds[str(ttt)] = np.array(ttemp)
        
        
                
    return out_spikes_tot,out_spikes_tot_mat,non_stimul_inds
#==============================================================================
#==============================================================================



