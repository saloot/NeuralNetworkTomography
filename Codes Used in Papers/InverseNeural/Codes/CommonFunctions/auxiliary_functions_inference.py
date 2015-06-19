#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
from brian import *
from scipy import sparse
import pdb
import random
import copy
import numpy.ma as ma
#==============================================================================

#==============================================================================
#========================THE BASIC INFERENCE ALGORITHM=========================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function truncates performs different inference methods.

# INPUT:
#    in_spikes:  the matrix containing information about spike times of pre-synaptic neurons
#    out_spikes: the matrix containing information about spike times of post-synaptic neurons
#    inference_method: '3' for STOCHASTIC NEUINF
#                      '4' for Cross Correlogram
#    inferece_params: a vector containing the parameters for each inference algorithm
#    W_estim: the 'fixed entries' in the association matrix: if an entry is not 'nan',
#             it means we are sure about its value and there is no need for the algorithm to worry about those entries.
#    location_flag: if '1' it means the algorithm is topology-aware
#    data_mode: 'R' for the general case with background traffic
#               'F' for the stimulate-and-observe case

#------------------------------------------------------------------------------
def inference_alg_per_layer(in_spikes,out_spikes,inference_method,inferece_params,W_estim,location_flag,data_mode):
    
    
    #------------------------------Initialization------------------------------
    n,TT = in_spikes.shape
    s = out_spikes.shape
    if len(s) >1:
        m = s[0]
    else:
        m = 1
    
    
    W_inferred = np.zeros([n,m])
    W_inferred.fill(0)
    
    Updated_Vals = np.zeros([n,m])
    
    cost = []
    W_estimated = copy.deepcopy(W_estim)            
    if norm(W_estimated):
        fixed_ind = 1-isnan(W_estimated).astype(int)
        fixed_ind = fixed_ind.reshape(n,m)
        W_inferred_orig = W_estimated
        W_inferred_orig = W_inferred_orig.reshape(n,m)
        for i in range(0,n):
            for j in range(0,m):
                if isnan(W_inferred_orig[i,j]):
                    W_inferred_orig[i,j] = W_inferred[i,j]
                    
        W_inferred = W_inferred_orig        
    else:
        fixed_ind = np.zeros([n,m])
    #--------------------------------------------------------------------------
    
    #----------------------The Perceptron-based Algorithm----------------------
    if (inference_method == 2) or (inference_method == 3):
        
        #......................Get Simulation Parameters.......................
        alpha0 = inferece_params[0]
        sparse_thr_0 = inferece_params[1]
        sparsity_flag = inferece_params[2]
        theta = inferece_params[3]
        max_itr_opt = inferece_params[4]        
        d_window = inferece_params[5]
        beta = inferece_params[6]        
        
        if len(inferece_params)>7:
            bin_size = inferece_params[7]
        else:
            bin_size = 0
        #......................................................................        
        
        #..............................Initializations.........................        
        range_tau = range(0,max_itr_opt)
        cost = np.zeros([len(range_tau)])
        neurons_ind = range(0,m)        
        #......................................................................
        
        #................Iteratively Update the Connectivity Matrix............
        if data_mode == 'R':
            
            
            for ijk in neurons_ind:                
                firing_inds = np.nonzero(out_spikes[ijk,:])
                firing_inds = firing_inds[0]
                
                print '-------------Neuron %s----------' %str(ijk2)
                
                for ttau in range_tau:
                    
                    #~~~~~~~~~~~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~~~~~~~~~
                    temp = 0
                    alpha = alpha0/float(1+log(ttau+1))
                    sparse_thr = sparse_thr_0/float(1+log(ttau+1))
                    fire_itr = -1
                    last_window = 0
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~Update in each Inter-Spike Time Interval~~~~~~~~~~~~~
                    for t_fire in firing_inds[:-1]:
                        fire_itr = fire_itr + 1
                        
                        for ttt in range(t_fire+1,min(firing_inds[fire_itr+1]+1,TT)):
                                
                            t_window = range(max(t_fire-d_window,last_window),ttt)
                            v = np.sign(sum(in_spikes[:,t_window],axis = 1))
                        
                            y_predict = 0.5*(1+np.sign(np.dot(W_inferred[:,ijk],v)-theta+0.00002))
                            
                            if y_predict:
                                if (ttt < firing_inds[fire_itr+1]-bin_size-1) and ( ttt> firing_inds[fire_itr] + bin_size):
                                    upd_val = np.multiply(v.T,np.random.randint(beta, size=n))
                                    W_inferred[:,ijk] = W_inferred[:,ijk] - alpha*np.multiply(upd_val,1-fixed_ind[:,ijk])
                                    cost[ttau] = cost[ttau] + 1
                                    last_window = ttt
                                    break
                            else:
                                if ttt >= firing_inds[fire_itr+1]-bin_size:
                                    upd_val = -np.multiply(v.T,np.random.randint(beta, size=n))
                                    W_inferred[:,ijk] = W_inferred[:,ijk] - alpha*np.multiply(upd_val,1-fixed_ind[:,ijk])
                                    cost[ttau] = cost[ttau] + 1
                                    last_window = 0
                                    break
                        
                        #~~~~~~~~~~~~~~~~~Apply Sparsity if NEcessary~~~~~~~~~~~~~~~~~~
                        if (sparsity_flag):
                            W_temp = soft_threshold(W_inferred.ravel(),sparse_thr)
                            W_inferred = W_temp.reshape([n,m])
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~~~~~~~~~Saturate the Weights~~~~~~~~~~~~~~~~~~~~~
                    ter = W_inferred < 0.001
                    W_inferred = np.multiply(ter.astype(int),W_inferred) + 0.001*(1-ter.astype(int))
                    ter = W_inferred >- 0.005
                    W_inferred = np.multiply(ter.astype(int),W_inferred) - 0.005*(1-ter.astype(int))
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        else:
            
            for ttau in range_tau:
                
                #~~~~~~~~~~~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~~~~~~~~~
                temp = 0
                alpha = alpha0/float(1+log(ttau+1))
                sparse_thr = sparse_thr_0/float(1+log(ttau+1))
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~Randomize Runs~~~~~~~~~~~~~~~~~~~~~~~~~
                shuffled_ind = range(0,TT)
                random.shuffle(shuffled_ind)
                neurons_ind = range(0,m)            
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
                #~~~~~~~~~~~~~~~~~~~~~~~~~~Perform Inference~~~~~~~~~~~~~~~~~~~~~~~
                for ttt in range(0,TT):
                    cascade_count = shuffled_ind[ttt]
                    x = in_spikes[:,cascade_count]
                    x = x.reshape(n,1)
                    if (location_flag == 1):
                        v = (x>0).astype(int)
                    y = out_spikes[:,cascade_count]
                    y = y.reshape(m,1)
                    yy_predict = np.zeros([m,1])
                    
                    random.shuffle(neurons_ind)
                    #~~~~~~~~~~~Upate the Incoming Weights to Each Neuron~~~~~~~~~~
                    if not_stimul_inds:
                        non_stim_inds = not_stimul_inds[str(cascade_count)]
                    else:
                        non_stim_inds = range(0,m)
                    for ijk2 in non_stim_inds:
                        #ijk = neurons_ind[ijk2]
                        ijk = ijk2
                        yy = y[ijk]                    
                        WW = W_inferred[:,ijk]
                        if (location_flag == 0):
                            if inference_method == 2:
                                v = (x>0).astype(int)
                            else:
                                if yy > 0:
                                    v = (x<yy).astype(int)
                                    v = np.multiply(v,(x>0).astype(int))
                                else:
                                    v = (x>0).astype(int)
                            
                    
                        y_predict = 0.5*(1+np.sign(np.dot(WW,v)-theta+0.00002))
                        #if abs(y_predict):
                        #    pdb.set_trace()
                        upd_val = np.dot(y_predict - np.sign(yy),v.T)
                        W_inferred[:,ijk] = W_inferred[:,ijk] - alpha*np.multiply(upd_val,1-fixed_ind[:,ijk])
                        Updated_Vals[:,ijk] = Updated_Vals[:,ijk] + np.sign(abs(v.T))
                        
                        cost[ttau] = cost[ttau] + sum(pow(y_predict - (yy>0.0001).astype(int),2))
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                    #~~~~~~~~~~~~~~~~~~~~~Saturate the Weights~~~~~~~~~~~~~~~~~~~~                
                    ter = W_inferred < 0.001
                    W_inferred = np.multiply(ter.astype(int),W_inferred) + 0.001*(1-ter.astype(int))
                    ter = W_inferred >- 0.005
                    W_inferred = np.multiply(ter.astype(int),W_inferred) - 0.005*(1-ter.astype(int))
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            
                #~~~~~~~~~~~~~~~~~~~Apply Sparsity if NEcessary~~~~~~~~~~~~~~~~~~~~
                if (sparsity_flag):
                    W_temp = soft_threshold(W_inferred.ravel(),sparse_thr)
                    W_inferred = W_temp.reshape([n,m])
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
                #~~~~~~~~~~~~~~~~~~~~~~Check Stopping Conditions~~~~~~~~~~~~~~~~~~~
                if (cost[ttau] == 0):
                    cost = cost[0:ttau+1]
                    break
                elif (ttau>300):
                    if ( abs(cost[ttau]-cost[ttau-2])/float(cost[ttau]) < 0.0001):
                        cost = cost[0:ttau+1]
                        break
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #------------------------------------------------------------------------------
    
    #-------------------------The Cross Correlogram Algorithm----------------------
    elif (inference_method == 4):
        d_window = inferece_params[0]           
        d_range = range(1,d_window+1)        
        D_estimated = np.zeros([n,m])
        W_temp = np.zeros([n,m])
        if data_mode == 'R':
            for i in range(0,n):
                in_sp_orig = in_spikes[i,:]
                mu_in = (in_sp_orig>0).astype(int)
                mu_in = mu_in.mean()
                for j in range(0,m):
                    out_sp = out_spikes[j,:]
                    mu_out = (out_sp>0).astype(int)
                    mu_out = mu_out.mean()
                    cc = np.zeros([len(d_range)])
                    #c0 = abs(float(sum(np.multiply(in_sp_orig>0,in_sp_orig == out_sp))-TT*mu_in*mu_out))
                    c0 = np.dot(in_sp_orig-mu_in,(out_sp-mu_out).T)
                    itr = 0
                    for d in d_range:    
                        in_sp = np.roll(in_sp_orig,d)
                        in_sp[0:d] = 0
                        cc[itr] = np.dot(in_sp-mu_in,(out_sp-mu_out).T)#/c0    
                        itr = itr + 1
                    
                    d_estim = d_range[np.argmax(cc)-1]
                    cd = diff(cc)
                    ii = np.argmax(cd)
                
                    D_estimated[i,j] = d_estim
                    if abs(cd).max()>0:
                        ii = cd.argmax()
                        if ii < len(cd)-1:
                            if cd[ii] > cd[ii + 1]:
                                W_temp[i,j] = abs(cc[ii+1]/abs(c0+0.001))
                            else:
                                W_temp[i,j] = -abs(cc[ii]/abs(c0+0.001))
                        else:
                            W_temp[i,j] = -abs(cc[ii]/abs(c0+0.001))
                
        else:
            for i in range(0,n):
                in_sp_orig = in_spikes[i,:]
                mu_in = (in_sp_orig>0).astype(int)
                mu_in = mu_in.mean()
                for j in range(0,m):
                    out_sp = out_spikes[j,:]
                    mu_out = (out_sp>0).astype(int)
                    mu_out = mu_out.mean()
                    cc = np.zeros([len(d_range)])
                    c0 = abs(float(sum(np.multiply(in_sp_orig>0,in_sp_orig == out_sp))-TT*mu_in*mu_out))
                    itr = 0
                    for d in d_range:    
                        in_sp = in_sp_orig + d/10000.0 #np.roll(in_sp_orig,d)
                        #in_sp[0:d] = 0
                        cc[itr] = (sum(np.multiply(in_sp_orig>0,in_sp == out_sp))-sum(np.multiply(in_sp>0,out_sp==0))-TT*mu_in*mu_out)#/float(c0) #np.dot(in_sp-mu_in,(out_sp-mu_out).T)/c0    
                        itr = itr + 1
                    
                
        
                    d_estim = d_range[np.argmax(cc)-1]
                    cd = diff(cc)
                    ii = np.argmax(cd)
                
                    D_estimated[i,j] = d_estim
                    if abs(cd).max()>0:
                        ii = cd.argmax()
                        if cd[ii] > cd[ii + 1]:
                            W_temp[i,j] = abs(cc[ii+1]/abs(c0+0.001))
                        else:
                            W_temp[i,j] = -abs(cc[ii]/abs(c0+0.001))
                    #W_temp[i,j] = np.dot(in_sp_orig,out_sp.T)
            
                                
        W_inferred = np.multiply(1-fixed_ind,W_temp) + np.multiply(fixed_ind,W_inferred)
    #--------------------------------------------------------------------------
    
    
#==============================================================================
#==============================================================================

