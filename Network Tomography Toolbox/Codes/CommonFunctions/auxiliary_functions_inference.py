#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
#from brian import *
from scipy import sparse,linalg
import pdb,os,sys
import random
import copy
import numpy.ma as ma
import matplotlib.pyplot as plt
import numpy as np
import math
from default_values import *
import time
from time import time
from numpy.random import randint
from numpy.random import RandomState
import linecache
#from scipy.optimize import minimize,linprog
try:
    from cvxopt import solvers, matrix, spdiag, log
    cvx_flag = 1
except:
    print 'CVXOpt is not installed. No biggie!'
    cvx_flag = 0

from scipy import optimize
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

def F(x=None,z=None,n=201):
    if x is None:
        return 0, matrix(np.zeros([n,1]))
    else:
        f = 0#np.sum(abs(x))
        Df = matrix(np.zeros([1,n]))#matrix(200*np.multiply(x.T,1-pow(np.tanh(100*pow(x.T,2)),2)))
        
        if z is None:  
            return f,Df
        else:
            #bb = 200*(1-pow(np.tanh(100*pow(x,2)),2))
            #aa = matrix(np.multiply(bb,1-200*np.multiply(pow(x,2),np.tanh(100*pow(x,2)))))
            aa = matrix(np.zeros([1,n]))
            H = spdiag(z[0] * aa)
            return f, Df, H
    #return np.linalg.norm(x)

def func_deriv(x):
    """ Derivative of objective function """
    return 2*x

def inference_alg_per_layer(in_spikes,out_spikes,inference_method,inferece_params,W_estim,location_flag,data_mode,range_neuron):
    
    from auxiliary_functions import soft_threshold
    
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
    if np.linalg.norm(W_estimated):
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
                
                print '-------------Neuron %s----------' %str(ijk)
                
                for ttau in range_tau:
                    
                    #~~~~~~~~~~~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~~~~~~~~~
                    temp = 0
                    alpha = alpha0/float(1+math.log(ttau+1))
                    sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
                    fire_itr = -1
                    last_window = 0
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~Update in each Inter-Spike Time Interval~~~~~~~~~~~~~
                    for t_fire in firing_inds[:-1]:
                        fire_itr = fire_itr + 1
                        
                        for ttt in range(t_fire+1,min(firing_inds[fire_itr+1]+1,TT)):
                                
                            t_window = range(max(t_fire-d_window,last_window),ttt)
                            v = np.sign(np.sum(in_spikes[:,t_window],axis = 1))
                        
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
                alpha = alpha0/float(1+math.log(ttau+1))
                sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
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
                    cd = np.diff(cc)
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
                    cd = np.diff(cc)
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
    
    #----------------------The MSE Cost-based Algorithm----------------------
    elif inference_method == 7:
        
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
        
        ss,TT = out_spikes.shape
        if len(range_neuron):
            neuron_range = range_neuron
        else:
            neuron_range = np.array(range(0,m))
            neuron_range = np.reshape(neuron_range,[m,1])
        #......................................................................        
        
        #..............................Initializations.........................        
        range_tau = range(0,max_itr_opt)
        cost = np.zeros([len(range_tau)])        
        #......................................................................
        
        #................Iteratively Update the Connectivity Matrix............
        if data_mode == 'R':
            ##CC = np.roll(out_spikes[:,],1, axis=1)
            CC = np.roll(out_spikes,1, axis=1)
            CC[:,0] = 0
            V = np.zeros([n,TT])
            for jj in range(1,TT-1):
                V[:,jj] = np.sum(CC[:,max(0,jj-d_window):jj-1],axis = 1)
                
            #V = np.sign(V)
            #pdb.set_trace()
            #V = CC
            
            
            
                
            #A = np.dot(V,V.T) + np.eye(n)
            #A_i = np.linalg.inv(A)
            #S = linalg.sqrtm(A)
            #S_i = np.linalg.inv(S)
            #B0 = np.dot(V,out_spikes.T)
            #pdb.set_trace()
            
            if sparsity_flag == 1:
                T_temp = TT-1
                range_T = range(T_temp,TT,T_temp)
                for T_T in range_T:
                    #Vv = np.zeros([n,T_temp])
                    #itr_j = 0
                    #for jj in range(T_T-T_temp,T_T):
                    #    Vv[:,itr_j] = np.sum(out_spikes[:,max(0,jj-d_window):jj],axis = 1)
                    #    itr_j = itr_j + 1
                    #pdb.set_trace()
                    
            
                    Vv = V[:,T_T-T_temp:T_T]
                    
                    A = np.dot(Vv,Vv.T) + np.eye(n)
                    A_i = np.linalg.inv(A)
                    S = linalg.sqrtm(A)
                    #S_i = np.linalg.inv(S)
                    B0 = np.dot(Vv,out_spikes[:,T_T-T_temp:T_T].T)
                    
                    print 'We are at %s' %str(T_T)
                    
                    for ijk in neuron_range:                
                        #Y = 10*out_spikes[ijk,T_T-T_temp:T_T] + 5 * np.ones([1,T_temp])
                        Y = out_spikes[ijk,T_T-T_temp:T_T]
                        
                        
                        print '-------------Neuron %s----------' %str(ijk)
                    
                        WW = W_inferred[:,ijk]
                        U = WW.T
                        
                        
                        for ttau in range_tau:
                    
                            #~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~                        
                            sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
                            #sparse_thr = sparse_thr_0/float(ttau+1)
                            
                            B = B0[:,ijk] + U.T#B_temp + U.T
                            #D = np.dot(B.T,S_i)
                            D = B.T
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            
                            #~~~~~~~~~Minimize with Respect to W~~~~~~~~~
                            W = np.dot(D,A_i)
                            
                            #W = np.linalg.lstsq(A,D.T)
                            #W = np.array(W[0])
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
                            #~~~~~~Apply Sparsity Regularizer to W~~~~~~~
                            U = soft_threshold(W,sparse_thr)
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            
                            #pdb.set_trace()
                            
                        #pdb.set_trace()
                        W_inferred[:,ijk] = W.T#np.vstack([W[0:ijk],0,W[ijk:]])
                    
                    
            elif sparsity_flag == 0 :
                
                T_temp = 2000
                #..............................Initializations.........................        
                range_tau = range(0,2)
                range_T = range(T_temp,TT,T_temp)
                #......................................................................
                
                for T_T in range_T:
                    #pdb.set_trace()     
                    Vv = V[:,T_T-T_temp:T_T].T
                    #--------
                    for ijk in neuron_range:                
                        print '-------------Neuron %s----------' %str(ijk)
                        
                        Y = out_spikes[ijk,T_T-T_temp:T_T].T
                        EE = np.delete(Vv.T,ijk,0).T
                        WW = np.linalg.lstsq(EE,Y)
                        WW = np.array(WW[0])
                        #pdb.set_trace()
                        W_inferred[:,ijk] = W_inferred[:,ijk] + np.vstack([WW[0:ijk],0,WW[ijk:]])
                    #--------
            
            elif sparsity_flag == 3 :
                T_temp = 1000
                #..............................Initializations.........................        
                range_tau = range(0,2)
                range_T = range(T_temp,TT,T_temp)
                #......................................................................
                
                for T_T in range_T:
                    Vv = V[:,T_T-T_temp:T_T]
                                    
                    for ijk in neuron_range:
                        Y = out_spikes[ijk,T_T-T_temp:T_T]
                        theta = 0.5
                        #aa = np.sum(np.multiply(Vv,np.dot(np.ones([n,1]),np.sign(Y-0.5))),axis=1)
                        #aa = np.reshape(aa,[n,1])
                        
                        #cons = ({'type': 'ineq',
                        #         'fun' : lambda x: np.multiply(np.sign(Y.T-0.5),(np.dot(Vv.T,x)-theta*np.ones([T_temp,1]))),
                        #         'jac' : lambda x: aa})
                        
                        
                        #res = minimize(func, np.zeros([n,1]), args=(), jac=func_deriv,constraints=cons, method='SLSQP', options={'disp': True})
                        c = np.zeros([n])
                        
                        A = np.dot(Vv,np.dot(np.ones([T_temp,1]),np.sign(Y-0.5)))
                        b = np.multiply(np.sign(Y-0.5).T,theta*np.ones([T_temp,1]))
                        
                        
                        res = linprog(c, A_ub=A.T, b_ub=b, bounds=(),options={"disp": True})
                        pdb.set_trace()
                        if res.success == 'True':
                            W_inferred[:,ijk] = W_inferred[:,ijk] + np.reshape(res.x,[n,1])
                        elif res.status<2:
                            W_inferred[:,ijk] = W_inferred[:,ijk] + np.reshape(res.x,[n,1])
                    
        
            
            
            elif sparsity_flag == 5:
                T_temp = TT-1#min(5000,TT-1)#TT-1
                range_T = range(T_temp,TT,T_temp)
                W_total = {}
                #print 'We are at %s' %str(T_T)
                if len(range_neuron) > 0:
                    neurons_range = range_neuron
                    
                for ijk in neuron_range:
                    print '-------------Neuron %s----------' %str(ijk)
                    for T_T in range_T:
                        Vv = V[:,T_T-T_temp:T_T]
                    
                        
                        Y = out_spikes[ijk,T_T-T_temp:T_T]
                        bb = np.nonzero(Y)
                        EE = Vv
                        if len(bb)>1:
                            bb = bb[1]
                        else:
                            bb = bb[0]
                            
                        
                        T1 = len(bb)
                        bb = np.reshape(bb,[1,T1])
                        V1 = EE[:,bb]
                        
                        bb = np.nonzero(1-Y)
                        if len(bb)>1:
                            bb = bb[1]
                        else:
                            bb = bb[0]
                        T2 = len(bb)
                        bb = np.reshape(bb,[1,T2])
                        V2 = EE[:,bb]
                        
                        
                        A = np.vstack([V1.T,-V2.T])
                        ff = theta*np.vstack([np.ones([T1,1]),-np.ones([T2,1])]) + 2.25*np.ones([T1+T2,1])
                        gamm = 1.25
                        
                        
                        C = np.eye(n) - gamm * np.dot(A.T,A)
                        C_i = np.linalg.inv(C)
                        #S = linalg.sqrtm(A)
                    
                        WW = W_inferred[:,ijk]
                        WW = np.reshape(WW,[n,1])
                        U = WW + np.random.rand(n,1)
                        b = np.dot(A.T,ff)
                        
                        for ttau in range_tau:
                    
                            #~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~                        
                            sparse_thr = sparse_thr_0/float(1+math.log(0.1*ttau+1))
                            #sparse_thr = sparse_thr_0/float(1+1*ttau)
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            
                            #~~~~~~~~~Minimize with Respect to W~~~~~~~~~
                            W = np.dot(C_i,(U-gamm*b))
                            #W = np.multiply(W,(W>0).astype(int))
                            #W = W/np.linalg.norm(W)
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
                            #~~~~~~Apply Sparsity Regularizer to W~~~~~~~
                            U = soft_threshold(W,sparse_thr)
                            #pdb.set_trace()
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            
                        if 0:
                            if (T_T == range_T[0]):
                                W_total[str(ijk)] = [U]
                            else:
                                #pdb.set_trace()
                                W_total[str(ijk)].append(U)
                    
                        
                        W_inferred[:,ijk] = W_inferred[:,ijk] + U.ravel()
                    
                    if 0 :
                        WW = W_total[str(ijk)]
                        W_a = []
                        for i in range(0,len(range_T)):
                            We = WW[i]
                            if i == 0:
                                Wa = We
                            else:
                                Wa = np.hstack([Wa,We])
                        
                        if min(Wa.shape)>1:
                            ee = Wa.mean(axis = 1)
                            ss = Wa.std(axis = 1)
                        else:
                            ee = Wa
                            ss = np.zeros([len(ee),1])
                        if ijk == 0:
                            pdb.set_trace()
                            
                        W_inferred[0:len(ee),ijk] = W_inferred[0:len(ee),ijk] + (np.multiply((ss<0.5*ss.mean()).astype(int),ee)).ravel()
                    #pdb.set_trace()
            
            else:
                
                T_temp = 1000
                #..............................Initializations.........................        
                range_tau = range(0,2)
                range_T = range(T_temp,TT,T_temp)
                #......................................................................
        
                #for ttau in range_tau:
                if 1:
                    for T_T in range_T:
                        Vv = np.zeros([n,T_temp])
                        itr_j = 0
                        for jj in range(T_T-T_temp,T_T):
                            Vv[:,itr_j] = np.sum(out_spikes[:,max(0,jj-d_window):jj],axis = 1)
                            itr_j = itr_j + 1
                        
                        #Vv = out_spikes[:,T_T-T_temp:T_T]
                        #pdb.set_trace()
                        print 'We are at %s' %str(T_T)
                        
                        E = np.linalg.pinv(Vv.T)
                        Y = out_spikes[:,T_T-T_temp:T_T].T #+ 0.00*np.ones([n,TT])                
                        WW2 = np.dot(E,Y)
                        #WW = np.linalg.lstsq(E.T,Y)
                        #pdb.set_trace()
                        for iil in range(0,n):
                            WW2[iil,iil] = 0
                        
                        
                        W_inferred = W_inferred + WW2
                        
                        


            

    else:
        print('Error! Invalid inference method.')
        sys.exit()
    
    return  W_inferred,cost,Updated_Vals
#==============================================================================
#==============================================================================


#==============================================================================
#==========================parse_commands_inf_algo=============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function runs the neural networks and generatethe required neural
# activity. The Brian simulator is used for this part.

# INPUTS:
#    input_opts: the options provided by the user
#------------------------------------------------------------------------------

def parse_commands_inf_algo(input_opts):
    neuron_range = []
    if (input_opts):
        for opt, arg in input_opts:
            if opt == '-Q':
                frac_stimulated_neurons = float(arg)                # Fraction of neurons in the input layer that will be excited by a stimulus
            elif opt == '-T':
                no_stimul_rounds = int(arg)                         # Number of times we inject stimulus to the network
            elif opt == '-S':
                ensemble_size = int(arg)                            # The number of random networks that will be generated                
            elif opt == '-A':
                file_name_base_data = str(arg)                      # The folder to store results
            elif opt == '-F':
                ensemble_count_init = int(arg)                      # The ensemble to start simulations from        
            elif opt == '-R':
                random_delay_flag = int(arg)                        # The ensemble to start simulations from                        
            elif opt == '-M':
                inference_method = int(arg)                         # The inference method
            elif opt == '-G':
                generate_data_mode = str(arg)                       # The data generating method            
            elif opt == '-Y':
                sparsity_flag = int(arg)                            # The flag that determines if sparsity should be observed during inference
            elif opt == '-X':
                infer_itr_max = int(arg)                            # The flag that determines if sparsity should be observed during inference            
            elif opt == '-K':
                we_know_topology = str(arg)                         # The flag that determines if we know the location of neurons (with respect to each other) (Y/N)
            elif opt == '-C': 
                pre_synaptic_method = str(arg)                      # The flag that determines if all previous-layers neurons count as  pre-synaptic (A/O)
            elif opt == '-V': 
                verify_flag = int(arg)                              # If 1, the post-synaptic states will be predicted
            elif opt == '-J': 
                delay_known_flag = str(arg)                         # If 'Y', we assume that the delay is known during the inference algorithm
            elif opt == '-U': 
                beta = int(arg)                                     # Specify the update probability paramter (p = 1/beta) in STOCHASTIC NEUINF
            elif opt == '-Z': 
                alpha0 = float(arg)                                 # Specify the update learnining rate
            elif opt == '-p': 
                p_miss = float(arg)                                 # The probability of missing a spike
            elif opt == '-j': 
                jitt = int(arg)                                     # Maximum  amount of randomjitter added to spike times (in miliseconds)
            elif opt == '-b': 
                bin_size = int(arg)                                 # If it is nonzero, the spikes will be placed within bins of size "bin_size"
            elif opt == '-o': 
                temp = (arg).split(',')                             # The range of neurons to identify the connections
                neuron_range = []
                for i in temp:                        
                    neuron_range.append(int(i))
            elif opt == '-h':
                print(help_message)
                sys.exit()
    else:
        print('Code will be executed using default values')
        
        
    #------------Set the Default Values if Variables are not Defines---------------
    if 'frac_stimulated_neurons' not in locals():
        frac_stimulated_neurons = FRAC_STIMULATED_NEURONS_DEFAULT
        print('ATTENTION: The default value of %s for frac_stimulated_neurons is considered.\n' %str(frac_stimulated_neurons))

    if 'infer_itr_max' not in locals():
        infer_itr_max = INFERENCE_ITR_MAX_DEFAULT
        print('ATTENTION: The default value of %s for infer_itr_max is considered.\n' %str(infer_itr_max))
        
    if 'no_stimul_rounds' not in locals():        
        no_stimul_rounds = NO_STIMUL_ROUNDS_DEFAULT
        print('ATTENTION: The default value of %s for no_stimul_rounds is considered.\n' %str(no_stimul_rounds))

    if 'ensemble_size' not in locals():            
        ensemble_size = ENSEMBLE_SIZE_DEFAULT
        print('ATTENTION: The default value of %s for ensemble_size is considered.\n' %str(ensemble_size))
    
    if 'file_name_base_data' not in locals():
        file_name_base_data = FILE_NAME_BASE_DATA_DEFAULT;
        print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_data))

    if 'ensemble_count_init' not in locals():
        ensemble_count_init = ENSEMBLE_COUNT_INIT_DEFAULT;
        print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))
    
    if 'ternary_mode' not in locals():
        ternary_mode = TERNARY_MODE_DEFAULT;
        print('ATTENTION: The default value of %s for ternary_mode is considered.\n' %str(ternary_mode))

    if 'file_name_base_results' not in locals():
        file_name_base_results = FILE_NAME_BASE_RESULT_DEFAULT;
        print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_results))

    if 'inference_method' not in locals():
        inference_method = INFERENCE_METHOD_DEFAULT;
        print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(inference_method))

    if 'sparsity_flag' not in locals():
        sparsity_flag = SPARSITY_FLAG_DEFAULT;
        print('ATTENTION: The default value of %s for sparsity_flag is considered.\n' %str(sparsity_flag))
    
    if 'generate_data_mode' not in locals():
        generate_data_mode = GENERATE_DATA_MODE_DEFAULT
        print('ATTENTION: The default value of %s for generate_data_mode is considered.\n' %str(generate_data_mode))

    if 'we_know_topology' not in locals():
        we_know_topology = WE_KNOW_TOPOLOGY_DEFAULT
        print('ATTENTION: The default value of %s for we_know_topology is considered.\n' %str(we_know_topology))

    if 'verify_flag' not in locals():
        verify_flag = VERIFY_FLAG_DEFAULT
        print('ATTENTION: The default value of %s for verify_flag is considered.\n' %str(verify_flag))
    
    if 'beta' not in locals():
        beta = BETA_DEFAULT
        print('ATTENTION: The default value of %s for beta is considered.\n' %str(beta))
    
    if 'alpha0' not in locals():
        alpha0 = ALPHA0_DEFAULT
        print('ATTENTION: The default value of %s for alpha0 is considered.\n' %str(alpha0))
        
    if 'p_miss' not in locals():
        p_miss = P_MISS_DEFAULT
        print('ATTENTION: The default value of %s for p_miss is considered.\n' %str(p_miss))
    
    if 'jitt' not in locals():
        jitt = JITTER_DEFAULT
        print('ATTENTION: The default value of %s for jitt is considered.\n' %str(jitt))
        
    if 'bin_size' not in locals():
        bin_size = BIN_SIZE_DEFAULT
        print('ATTENTION: The default value of %s for bin_size is considered.\n' %str(bin_size))
    #------------------------------------------------------------------------------

    #------------------Create the Necessary Directories if Necessary---------------
    if not os.path.isdir(file_name_base_results):
        os.makedirs(file_name_base_results)    
    if not os.path.isdir(file_name_base_results+'/Inferred_Graphs'):
        temp = file_name_base_results + '/Inferred_Graphs'
        os.makedirs(temp)
    if not os.path.isdir(file_name_base_results+'/Accuracies'):
        temp = file_name_base_results + '/Accuracies'
        os.makedirs(temp)            
    if not os.path.isdir(file_name_base_results+'/Plot_Results'):    
        temp = file_name_base_results + '/Plot_Results'
        os.makedirs(temp)    
    #------------------------------------------------------------------------------


    return frac_stimulated_neurons,no_stimul_rounds,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,file_name_base_results,inference_method,sparsity_flag,we_know_topology,verify_flag,beta,alpha0,infer_itr_max,p_miss,jitt,bin_size,neuron_range
#==============================================================================
#==============================================================================

#==============================================================================
#=============================extract_high_SNR_t===============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function ideintifies the time instances with possibly high SNR values
# which we can use in the inference algorithm.

#*******ISSUE TO FIX******
# These high SNR values should in principle be "consecutive"!
# Thus we need some sort of solution, may be randomly sample some of them?
#*************************

# INPUTS:
#   spikes_times_mat: the matrix containing the firing activity of neurons
#   output_spikes_times: the output spiking times (for the post-synaptic neuron)
#   max_no_values: maximum number of high SNR values to return

# OUTPUTS:
#   spikes_SNR: high SNR values
#   ind_t: the index of time instances where SNR was high
#------------------------------------------------------------------------------

def extract_high_SNR_t(spikes_times_mat,max_no_values,output_spikes_times):
    
    #--------------Sort the Incoming Signals and Pick the Best-----------------
    ind_t = np.argsort(np.sum(spikes_times_mat,axis = 1)    )
    ind_t = ind_t[-max_no_values:]
    spikes_SNR = spikes_times_mat[ind_t,:]
    #--------------------------------------------------------------------------
    
    
    #--Include the Entries Corresponding to When Output Neuron Fired As Well---
    bb = np.nonzero(output_spikes_times)
    bb = bb[0]
    
    spikes_SNR = np.vstack([spikes_SNR,spikes_times_mat[bb,:]])
    
    bb = np.reshape(bb,[len(bb),1])
    ind_t = np.reshape(ind_t,[len(ind_t),1])
    
    ind_t = np.vstack([ind_t,bb])
    ind_t = ind_t.ravel()
    #--------------------------------------------------------------------------
    
    return spikes_SNR,ind_t
#==============================================================================
#==============================================================================

#==============================================================================
#=============================perform_integration==============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function applies leaky integrations to a set of given spikes and saves
# them to the specified files for later use.
# To be able to handle large matrices, it divides the matrix into blocks of 
# size 'block_size' and perform integration on each block one at a time.

# INPUTS:
#   spikes_times: the matrix containing the firing activity of neurons
#   tau_d: the first leak constant (the membrane decay time constant)
#   tau_s: the first leak constant (the membrane rise time constant)
#   d_window: the time window to perform integration over
#   t_fire: the firing times of the neuron which we want to infer its incoming connections
#   file_name_base: the file name identifier which will be used to store the integration results

# OUTPUTS:
#   None
#------------------------------------------------------------------------------

def perform_integration(spikes_times,tau_d,tau_s,d_window,t_fire,file_name_base):
    
    #-----------------------------Initializations------------------------------
    T = spikes_times.shape[1]
    n = spikes_times.shape[0]
    dl = 0
    block_size = 40000
    TT_last = 0
    range_T = range(block_size,T,block_size)
    if T not in range_T:
        range_T.append(T)
    #--------------------------------------------------------------------------
    
    #-----------------Divide Into Smaller Blocks and Inetgrate-----------------
    for TT in range_T:
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~Read the Current Block~~~~~~~~~~~~~~~~~~~~~~~
        CC = spikes_times[:,TT_last:TT]
        V = np.zeros([n,TT-TT_last])
        X = np.zeros([n,TT-TT_last])
        t_last = 0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~~Create the Integration Window~~~~~~~~~~~~~~~~~~~~~
        AA = np.reshape(np.array(range(1,TT-TT_last+1)),[TT-TT_last,1])
        AA = np.dot(AA,np.ones([1,n]))
        AA = np.multiply(CC,AA.T)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~~~~Perform the Integration~~~~~~~~~~~~~~~~~~~~~~~~~
        for jj in range(0,TT-TT_last):
            
            t_min = max(0,t_last-d_window)                
            DD = AA[:,t_min:jj-dl]
                
            V[:,jj] = np.sum(np.multiply(np.sign(DD),np.exp(-(jj-DD-dl)/tau_d)),axis = 1)
            X[:,jj] = np.sum(np.multiply(np.sign(DD),np.exp(-(jj-DD-dl)/tau_s)),axis = 1)
            if jj in t_fire:
                t_last = jj
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~Write the Matrices Onto the File~~~~~~~~~~~~~~~~~~~~
        file_name_v = file_name_base + '_tau_d_' + str(int(tau_d)) + '.txt'
        integrate_file = open(file_name_v,'a+')
        for i in range(0,TT-TT_last):
            aa = np.reshape(V[:,i],[1,n])
            np.savetxt(integrate_file,aa,fmt="2.3%f")
        integrate_file.close()
        
        file_name_x = file_name_base + '_tau_s_' + str(int(tau_s)) + '.txt'
        integrate_file = open(file_name_x,'a+')
        for i in range(0,TT-TT_last):
            aa = np.reshape(X[:,i],[1,n])
            np.savetxt(integrate_file,aa,fmt="2.3%f")
        integrate_file.close()

        TT_last = TT
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    #--------------------------------------------------------------------------    

    return 1
#==============================================================================
#==============================================================================



def read_integration_matrix(file_name,start_line,end_line,n):
    
    import linecache
    
    V = np.zeros([end_line - start_line,n])
    
    for i in range(start_line,end_line):
        a = linecache.getline(file_name, i)
        if a:
            a = (a[:-2]).split(' ')
            a = np.array(a)
            
            a = a.astype(float)
            #a = np.reshape(a,[len(a),1])
            
            V[i-start_line,:] = a
            
        else:
            break
    
    return V

#==============================================================================
#==============================================================================


def read_spikes_lines(file_name,line_no,n):
    
    a = linecache.getline(file_name, line_no)
    if a:
        a = (a[:-1]).split(' ')
        a = np.array(a)
        if len(a[0]):
            a = a.astype(float)
        else:
            a = []
            
        return list(a)
            
    else:
        return []
    
    
    
def read_spikes_lines_delayed(file_name,line_no,n,d_max,dd):
    v = []
    if len(dd):
        aa = np.nonzero(dd)[0]
    else:
        aa = np.array(range(0,n)).astype(int)
        dd = np.ones([n]).astype(int)
    
    for i in aa:
        a = linecache.getline(file_name, max(0,line_no-dd[i]))
        if a:
            a = (a[:-1]).split(' ')
            a = np.array(a)
            if len(a[0]):
                a = a.astype(float)
            else:
                a = []
            
            if i in a:    
                v.append(i)
    
    #pdb.set_trace()
    return v

#==============================================================================
#==============================================================================



#==============================================================================
#=======================delayed_inference_constraints==========================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function runs the inference algorithm for the algorithm where delay and
# connectivity are co-optimized together. The model is based on the LIF neural
# model and approximates the membrane voltage kernel function as a double
# exponential whose time constants are specified within the code.

# INPUTS:
#   spikes_times: the matrix containing the firing activity of neurons
#   d_window: the time window to perform integration over
#   max_itr_opt: the maximum number of iterations to perform the optimization
#   sparse_thr_0: the initial sparsity threshold
#   theta: the firing threshold
#   W_act: the actual connectivity matrxi (for the DEVELOPMENT PHASE)
#   D_act: the actual delay matrxi (for the DEVELOPMENT PHASE)
#   neuron_range: the range of neurons to find the (incoming) connections of.
#                 If left empty ([]), the optimization will be performed for all neurons.

# OUTPUTS:
#   W_inferred: the inferred connectivity matrix
#   D_inferred: the inferred delay matrix
#------------------------------------------------------------------------------
def delayed_inference_constraints(spikes_times,d_window,max_itr_opt,sparse_thr_0,theta,W_act,D_act,neuron_range):
    
    #----------------------------Initilizations--------------------------------    
    from auxiliary_functions import soft_threshold
    import os.path
        
        
    n,TT = spikes_times.shape
    m = n
    W_inferred = np.zeros([n,m])
    D_inferred = np.zeros([n,m])
    
    range_tau = range(0,max_itr_opt)
    
    if len(neuron_range) == 0:
        neuron_range = np.array(range(0,m))
    
    gamm = 1                                       # Determine how much sparsity is important for the algorithm (the higher the more important)
    
    dl = 0#
    
    T_temp = min(40000,TT-1) #TT-1#            # Decide if the algorithm should divide the spike times into several smaller blocks and merge the results
    range_T = range(2*T_temp,TT,T_temp)
    W_total = {}
    D_total = {}
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV)
    t0 = math.log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 2/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    file_name_integrated_spikes_base = '../Data/Spikes/Moritz_Integrated_750' 
    #--------------------------------------------------------------------------
    
    
    #---------------Preprocess Spike Times and the Integration Effect----------
    CC = np.roll(spikes_times,1, axis=1)           # Shift the spikes time one ms to account for causality and minimum propagation delay
    #--------------------------------------------------------------------------    
    
    
    #---------Identify Incoming Connections to Each Neuron in the List---------
    for ijk in neuron_range:
        
        print '-------------Neuron %s----------' %str(ijk)
        Y = spikes_times[ijk,:]
        t_fire = np.nonzero(Y)
        t_fire = t_fire[0]
        t_last = 0
    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Experimental Block~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~Shift the Spikes When Delays Are Known~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (len(D_act)) and (0):
            dd = D_act[:,ijk]
            CC = np.zeros([n,TT])
            for i in range(0,n):
                CC[i,:] = np.roll(spikes_times[i,:],int(dd[i])+1)
                CC[i,0:int(dd[i])+1] = 0
            
            AA = np.reshape(np.array(range(1,TT+1)),[TT,1])
            AA = np.dot(AA,np.ones([1,n]))
            AA = np.multiply(CC,AA.T)
            d_window = 0
        if len(W_act) and len(D_act):    
            w = W_act[:,ijk]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Experimental Block~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~Extent the Integration Window to the Last Reset~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        file_name_integrated_spikes = file_name_integrated_spikes_base + '_' + str(ijk) + '_tau_d_' + str(int(tau_d)) + '.txt'
        file_name_integrated_spikes_base_ij = file_name_integrated_spikes_base + '_' + str(ijk)
        
        if not os.path.isfile(file_name_integrated_spikes):
            perform_integration(CC,tau_d,tau_s,d_window,t_fire,file_name_integrated_spikes_base_ij)
        
        file_name_V = file_name_integrated_spikes_base_ij + '_tau_d_' + str(int(tau_d)) + '.txt'
        
        
        #V = (np.genfromtxt(file_name_V, dtype=float)).T
        
           
        file_name_X = file_name_integrated_spikes_base_ij + '_tau_s_' + str(int(tau_s)) + '.txt'
        #X = (np.genfromtxt(file_name, dtype=float)).T
        
        
        #if 1:
            #for jj in range(dl,TT-1):
                
            #    t_min = max(0,t_last-d_window)                
            #    DD = AA[:,t_min:jj-dl]
                
            #    V[:,jj] = np.sum(np.multiply(np.sign(DD),np.exp(-(jj-DD-dl)/tau_d)),axis = 1)
            #    X[:,jj] = np.sum(np.multiply(np.sign(DD),np.exp(-(jj-DD-dl)/tau_s)),axis = 1)
            #    #U[:,jj] = V[:,jj]-X[:,jj]
            #    #if len(W_act) and len(D_act):
            #    #    H[ijk,jj] = U0*np.dot(V[:,jj]-X[:,jj],w)
            #    if jj in t_fire:
            #        t_last = jj
                    
        
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            
        
        #~~~~~~~~~~~~~~~~~Infer the Connections for Each Block~~~~~~~~~~~~~~~~~
        for T_T in range_T:
            
            #...................In-loop Initializations........................
            Y = spikes_times[ijk,T_T-T_temp:T_T]                    # Output firing pattern
            t_fire = np.nonzero(Y)
            t_fire = t_fire[0]
                        
            DD1 = math.exp(1.5/tau_d)*np.eye(n)                          # The first diagona delay matrix (corresponding to the decay coefficient)
            DD2 = math.exp(1.5/tau_s)*np.eye(n)                          # The second diagona delay matrix (corresponding to the rise coefficient)
            
            WW = W_inferred[:,ijk]
            Z = np.reshape(WW,[n,1])                                # The auxiliary optimization variable
            
            
            #IM = create_integration_matrix(t_fire,IM_base_2,d_i)
            #IM2 = create_integration_matrix(t_fire,IM_base_3,d_i)
            #CC_2 = CC[:,T_T-T_temp:T_T]
            #V = (np.dot(IM,CC_2.T)).T
            #X = (np.dot(IM2,CC_2.T)).T
            #..................................................................
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Experimental Block~~~~~~~~~~~~~~~~~~~~~~~~
            #~~~~~Some Tests to Verify the Algorithm When Connectivity Is Known~~~~
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if len(W_act) and len(D_act):
                dd = D_act[:,ijk]
                #pdb.set_trace()
                #dd = np.reshape(dd,[n,1])
                dd1 = np.multiply(np.sign(dd),np.exp(dd/(tau_d)))
                dd2 = np.multiply(np.sign(dd),np.exp(dd/(tau_s)))
                
                DD1 = np.diag(dd1)
                DD2 = np.diag(dd2)
                #w = W_act[:,ijk]
                #w = np.reshape(w,[n,1])
                
                #pdb.set_trace()
                #U = np.dot(V.T,DD1) - np.dot(X.T,DD2)
                
                #U = np.multiply(U,(U>0).astype(int))
                #U = np.multiply(U,(U<1/U0).astype(int)) + np.multiply(1/U0,(U>1/U0).astype(int))
                
                U = V.T-X.T
                
                H = U0*np.dot(U,w)
                
                hh = (H>theta).astype(int)
                #pdb.set_trace()
                #plt.plot(hh[0:100]);plt.plot(Y[0:100],'r');plt.show()
                #plt.plot(H[0:300]);plt.plot(0.1*Y[0:300],'r');plt.show()
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            
            #.................Read Integration Chunks from File....................
            V = read_integration_matrix(file_name_V,T_T-T_temp,T_T,n)
            X = read_integration_matrix(file_name_X,T_T-T_temp,T_T,n)
            #......................................................................
            
            #..........Construct the Positive and Zero State Matrices..........            
            bb = np.nonzero(Y)
            
            VV = V.T#[:,T_T-T_temp:T_T]
            XX = X.T#[:,T_T-T_temp:T_T]
            
            V1 = VV[:,bb[0]]
            X1 = XX[:,bb[0]]
            T1 = len(bb[0])
            bb = np.nonzero(1-Y)
            
            V2 = VV[:,bb[0]]
            X2 = XX[:,bb[0]]
            T2 = len(bb[0])
                
            U = np.vstack([V1.T,-V2.T])
            XX = np.vstack([X1.T,-X2.T])
            #..................................................................
            
            #........Pre-compute Some of Matrices to Speed Up the Process......
            B = np.hstack([U,-XX])
            #B_i = np.linalg.pinv(B)
            g = ((theta-h0)*np.vstack([np.ones([T1,1]),-np.ones([T2,1])]) + 5.55*np.ones([T1+T2,1]))/U0
            #U_i =  np.linalg.pinv(U)
            #..................................................................
                
    
            #................Infer the Connections for This Block..............
            if (len(W_act) == 0) and (len(D_act) == 0):
                
                #=============Optimize for Both Weights and Delay==============
                if 0:
                    for iau in range(0,500):
                        aa = np.vstack([DD1,DD2])
                        A = np.dot(B,aa)
                        C = gamm * np.eye(n) - np.dot(A.T,A)
                        C_i = np.linalg.inv(C)
                        
                        #=============Optimize for Weight First================                        
                        for ttau in range_tau:
                            
                            #~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~~
                            sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
                            b = gamm * Z - np.dot(A.T,g)
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    
                            #~~~~~~~~~~~Minimize with Respect to W~~~~~~~~~~~
                            W = np.dot(C_i,b)
                            #W = np.multiply(W,(W>0).astype(int))
                            #W = W/np.linalg.norm(W)
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
                            #~~~~~~~~Apply Sparsity Regularizer to W~~~~~~~~~
                            Z = soft_threshold(W,sparse_thr)
                            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        #======================================================
                        
                        #=================Now Iterate Over Delay===============
                        zz = np.multiply(np.sign(Z[:,0]),np.divide(1,Z[:,0] + 1e-10))       #Invert the matrix W*
                            
                        zz = np.reshape(zz,[n,1])
                        zz = np.vstack([zz,zz])
                        WW = np.diag(zz[:,0])                        
                        #WW = np.vstack([zz,zz])
                            
                        dd = -np.dot(np.dot(WW.T,B_i),g)
                        dd = 1e-15 + np.multiply(dd,dd>0)
                        dd1 = tau_d*np.log(dd[0:n,0])
                        dd2 = tau_s*np.log(dd[n:,0])
                        dd1 = np.multiply(dd1,dd1>0)+1
                        dd2 = np.multiply(dd2,dd2>0)+1
                        
                        #dd = (dd1 + dd2)/2.0
                        DD1 = np.diag(np.exp(dd1/tau_d))
                        DD2 = np.diag(np.exp(dd2/tau_s))
                        dd = np.reshape(dd2+dd1,[n,1])
                        #======================================================  

                    #====================Save the Results======================
                    if (T_T == range_T[0]):
                        W_total[str(ijk)] = [W]
                        D_total[str(ijk)] = [dd]
                    else:
                        #pdb.set_trace()
                        W_total[str(ijk)].append(W)
                        D_total[str(ijk)].append(dd1)
                    #==========================================================
                #==============================================================
                    
                #=================Optimize Only for Weights====================
                else:
                    print 'hello!'
                    #AA = np.dot(U.T,U)
                    #for i in range(0,n):
                    #    AA[i,i] = 0
                    #C = gamm * np.eye(n) - AA
                    #aa = np.vstack([DD1,DD2])
                    #A = np.dot(B,aa)
                    A = U
                    C = gamm * np.eye(n) - np.dot(A.T,A)
                    C_i = np.linalg.inv(C)
                        
                    for ttau in range_tau:
                        
                        #~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~                    
                        sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
                        #b = Z - gamm *  np.dot(U.T,g)
                        b = Z - gamm *  np.dot(A.T,g)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                
                        #~~~~~~~~~~~Minimize with Respect to W~~~~~~~~~~~
                        W = np.dot(C_i,b)
                        W = W/np.linalg.norm(W)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
                        #~~~~~~~~Apply Sparsity Regularizer to W~~~~~~~~~
                        Z = np.reshape(soft_threshold(W,sparse_thr),[n,1])
                        #pdb.set_trace()
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        
                    #====================Save the Results======================    
                    if (T_T == range_T[0]):
                        W_total[str(ijk)] = [W]
                        D_total[str(ijk)] = [W]
                    else:
                        #pdb.set_trace()
                        W_total[str(ijk)].append(W)
                        D_total[str(ijk)].append(W)
                    #==========================================================
                #==============================================================
            #..................................................................
            
            #..................................................................
            #.........................Experimental Block.......................
            #...Some Tests to Verify the Algorithm When Connectivity Is Known..
            #..................................................................
            else:
                    
                if 0:#len(W_act):
                    W = 1*(np.reshape(W_act[:,ijk],[n,1]))
                    #U_i = np.linalg.pinv(np.dot(XX-U,np.diag(W[:,0])))
                    U_i = np.linalg.pinv(np.dot(U,np.diag(W[:,0])))
                        
                    aa = np.dot(np.dot(U-XX,np.diag(W[:,0])),np.ones([n,1]))
                        
                    #dd = np.dot(U_i,g+1*aa)
                    dd = -np.dot(B_i,g)
                            
                    #for i in range(0,n):
                    #    if abs(W[i]) > 0:
                    #        dd[i] =dd[i]/W[i]
                    #    else:
                    #        dd[i] = 0
                    #dd = np.multiply(dd,pow(W+0.00001*np.random.rand(n,1),-1))
                    #dd = np.multiply(dd,(dd>0).astype(int))
                    pdb.set_trace()
                    #D_inferred[:,ijk] = dd[:,0]
                    if (T_T == range_T[0]):
                        D_total[str(ijk)] = [dd]
                    else:
                        D_total[str(ijk)].append(dd)
                else:
                    dd = D_act[:,ijk]
                    dd1 = np.multiply(np.sign(dd),np.exp(dd/(tau_d)))
                    dd2 = np.multiply(np.sign(dd),np.exp(dd/(tau_s)))
                    DD1 = np.diag(dd1)
                    DD2 = np.diag(dd2)
                    #A = (np.dot(U,DD1)-np.dot(XX,DD2))#np.dot(C,np.dot(UU,DD))
                    A = np.dot(U,DD1)
                    gamm = 5
                    C = gamm * np.eye(n) - np.dot(A.T,A)
                    C_i = np.linalg.inv(C)
                    for ttau in range(0,100):#range_tau:
                        
                        #~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~                        
                        sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
                        b = gamm * Z - np.dot(A.T,g)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                
                        #~~~~~~~~~Minimize with Respect to W~~~~~~~~~
                        W = np.dot(C_i,b)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
                        #~~~~~~Apply Sparsity Regularizer to W~~~~~~~
                        Z = soft_threshold(W,sparse_thr)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    #pdb.set_trace()    
                    if (T_T == range_T[0]):
                        W_total[str(ijk)] = [W]
                    else:
                        W_total[str(ijk)].append(W)
                    
            #..................................................................
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~Merge the Results of Different Blocks~~~~~~~~~~~~~~~~~
        
        if 1:#len(W_act)  == 0:           
            WW = W_total[str(ijk)]
            W_a = []
            for i in range(0,len(range_T)):
                We = WW[i]
                if i == 0:
                    Wa = We
                else:
                    Wa = np.hstack([Wa,We])
                                
            ee = Wa.mean(axis = 1)
            ss = Wa.std(axis = 1)
            uu = pow(ss,2)-pow(ee,2)
            #pdb.set_trace()           
            #W_inferred[:,ijk] = W_inferred[:,ijk] + np.multiply((abs(uu)>abs(uu.mean())+uu.mean()).astype(int),ee)
            W_inferred[:,ijk] = W_inferred[:,ijk] + np.multiply((ss<0.25*uu.mean()).astype(int),ee)
            W_inferred[:,ijk] = W_inferred[:,ijk] + ee 
            pdb.set_trace()
        if len(D_act)  == 0:
            DD = D_total[str(ijk)]
            D_a = []
            for i in range(0,len(range_T)):
                De = DD[i]
                if i == 0:
                    Da = De
                else:
                    Da = np.hstack([Da,De])
                
                
            ee = Da.mean(axis = 1)
            ss = Da.std(axis = 1)
            uu = pow(ss,2)-pow(ee,2)
            #D_inferred[:,ijk] = D_inferred[:,ijk] + np.multiply((uu>0.0001).astype(int),ee)
                
            D_inferred[:,ijk] = D_inferred[:,ijk] + ee
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            
    return W_inferred,D_inferred


def loss_func(x):
    #return np.dot(x.T, x)
    return sum(np.abs(x))

def jac(x):
    return 2 * np.sign(x)


def loss_func_lambda(x,FF,delta,AA,Z):
    
    #E = 0.25 * np.dot(np.dot(x.T,FF),x) - delta *sum(x)
    E = 0.25 * np.dot(np.dot(x.T,FF),x) - delta *sum(x) + np.dot(np.dot(x.T,AA),Z)
    return E
    
    
def jac_lambda(x,FF,delta,AA,Z):
    #return 0.5 * np.dot(FF,x) - delta * np.ones([len(x)])
    return 0.5 * np.dot(FF,x) - delta * np.ones([len(x)]) + np.dot(AA,Z).ravel()

#==============================================================================
#=======================delayed_inference_constraints==========================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function runs the inference algorithm for the algorithm where delay and
# connectivity are co-optimized together. The model is based on the LIF neural
# model and approximates the membrane voltage kernel function as a double
# exponential whose time constants are specified within the code.

# INPUTS:
#   spikes_times: the matrix containing the firing activity of neurons
#   d_window: the time window to perform integration over
#   max_itr_opt: the maximum number of iterations to perform the optimization
#   sparse_thr_0: the initial sparsity threshold
#   theta: the firing threshold
#   W_act: the actual connectivity matrxi (for the DEVELOPMENT PHASE)
#   D_act: the actual delay matrxi (for the DEVELOPMENT PHASE)
#   neuron_range: the range of neurons to find the (incoming) connections of.
#                 If left empty ([]), the optimization will be performed for all neurons.

# OUTPUTS:
#   W_inferred: the inferred connectivity matrix
#   D_inferred: the inferred delay matrix
#------------------------------------------------------------------------------
def delayed_inference_constraints_memory(out_spikes_tot_mat_file,TT,n,max_itr_opt,sparse_thr_0,alpha0,theta,neuron_range,W_act,DD_act):
    
    #----------------------------Initilizations--------------------------------    
    from auxiliary_functions import soft_threshold
    import os.path
        
        
    #n,TT = spikes_times.shape
    m = n
    W_inferred = np.zeros([n,m])
    D_inferred = np.zeros([n,m])
    
    range_tau = range(0,max_itr_opt)
    
    if len(neuron_range) == 0:
        neuron_range = np.array(range(0,m))
    
    gamm = 1                                       # Determine how much sparsity is important for the algorithm (the higher the more important)
    
    dl = 0#
    
    T0 = 20000                                  # It is the offset, i.e. the time from which on we will consider the firing activity
    T_temp = 20000                              # The size of the initial batch to calculate the initial inverse matrix
    range_T = range(T_temp+T0,TT)
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV)
    lamda = 1.0                                    # This is the "forgetting" factor
    eta = 0.5                                       # The tanh coefficient to approximate the sign function
    d_max = 10
    lamda_i = 1.0/lamda
    t_avg = 5
    block_size = 8000
    
    t0 = math.log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 2/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    #--------------------------------------------------------------------------
    
    #---------Identify Incoming Connections to Each Neuron in the List---------
    for ijk in neuron_range:
        
        print '-------------Neuron %s----------' %str(ijk)
    
        #DD1 = math.exp(1.5/tau_d)*np.eye(n)                          # The first diagona delay matrix (corresponding to the decay coefficient)
        #DD2 = math.exp(1.5/tau_s)*np.eye(n)                          # The second diagona delay matrix (corresponding to the rise coefficient)
        Z = np.reshape(W_inferred[:,ijk],[n,1])                                # The auxiliary optimization variable
    
        
        #~~~~~~~~~~~~~~~~~Calculate The Initial Inverse Matrix~~~~~~~~~~~~~~~~~
        X = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        V = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        x = np.zeros([n+1,1])
        v = np.zeros([n+1,1])
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        Y = np.zeros([1+int(T_temp/float(t_avg))])
        dd = []
        yy = 0
        
        t_counter = 0
        t_tot = 0
        for t in range(T0,T0 + T_temp):
            
            #........Pre-compute Some of Matrices to Speed Up the Process......
            #fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            if (ijk in fire_t):                
                yy = yy + 1
                x = np.zeros([n+1,1])
                v = np.zeros([n+1,1])
            
            
            #fire_t = read_spikes_lines_delayed(out_spikes_tot_mat_file,t,n,d_max,dd)
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
            x = math.exp(-1/tau_s) * x
            x[fire_t] = x[fire_t] + 1
            
            v = math.exp(-1/tau_d) * v
            v[fire_t] = v[fire_t] + 1
            
            if ((t % t_avg) == 0) and (t_counter):
                vv = vv/float(t_counter)
                xx = xx/float(t_counter)
                
                V[:,t_tot] = vv.ravel()
                X[:,t_tot] = xx.ravel()
                Y[t_tot] = yy
                
                xx = np.zeros([n+1,1])
                vv = np.zeros([n+1,1])
                yy = 0
                t_counter = 0
                t_tot = t_tot + 1
            else:
                vv = vv + v
                xx = xx + x
                vv[-1,0] = 1
                t_counter = t_counter + 1
        
        Y = np.array(Y)
        
        V = V[:,0:t_tot]
        X = X[:,0:t_tot]
        Y = Y[0:t_tot]
        
        g = (Y>0).astype(int) - (Y<=0).astype(int)
        A = (V-X).T
        #A = (V).T
        
        gg = theta * np.dot(A.T,g)
        #C = gamm * np.eye(n) - np.dot(A.T,A)
        #m = int(2*n)
        #S = np.random.randn(m,T_temp)
        #A = np.dot(S,A)
        
        AA = np.dot(np.diag(g.ravel()),A)
        AA = np.delete(AA.T,ijk,0).T
        TcT = len(g)
        delta = .005
        bc = delta*np.ones([TcT])
        #cons = {'type':'ineq','fun':lambda x: bc - np.dot(AA,x),'jac':lambda x: -AA}
        
        cons = ({'type':'ineq','fun':lambda x: np.dot(AA,x).ravel() - bc })

        opt = {'disp':False,'maxiter':500}
        W0 = np.random.randn(n,1)
        W0 = W0/np.linalg.norm(W0)
        
        #res_cons = optimize.minimize(loss_func, W0, jac=jac,constraints=cons,method='SLSQP', options=opt)
        #ww = np.reshape(res_cons['x'],[n,1])
        #W = np.zeros([n+1,1])
        #W[0:ijk,0] = ww[0:ijk,0]
        #W[ijk+1:,0] = ww[ijk:,0]
                
        lambda_0 = np.zeros([TcT,1])
        FF = np.dot(AA,AA.T)
        
        if 0:
            if 0:
                AC =  matrix(0.5 * FF)
                bbc = matrix(-bc)
                G = matrix(-np.eye(TcT))
                hh = matrix(np.zeros([TcT]))
                lambda_0 = matrix(np.zeros([TcT,1]))
                dims = {'l': TcT, 'q': [], 's': []}
                initvals = {'x':lambda_0}
                lam = solvers.coneqp(AC, bbc, G, hh, dims,initvals=initvals)['x']
                initvals = {'x':lam}
                
                Ab = matrix(AA[:,ijk])
                cb = matrix(np.zeros([1,1]))
                lam = solvers.coneqp(AC, bbc, G, hh, dims,Ab.T,cb,initvals=initvals)['x']
                lam = np.array(lam)
                W = np.dot(AA.T,lam)
            else:
                AC =  matrix(np.eye(n))
                bbc = matrix(np.zeros([n]))
                G = np.delete(AA.T,ijk,0).T
                G = matrix(G)
                hh = matrix(-np.reshape(bc,[TcT,1]))
                
                initvals = {'x':matrix(W0)}
                dims = {'l': TcT, 'q': [], 's': []}
                ww = solvers.coneqp(AC, bbc, G, hh)['x']
                ww = np.array(ww)
                W = np.zeros([n+1,1])
                W[0:ijk,0] = ww[0:ijk,0]
                W[ijk+1:,0] = ww[ijk:,0]
                
            
            pdb.set_trace()
        else:
            aa = np.ones([TcT,2])
            aa[:,0] = 0
            aa[:,1] = 100000
            bns = list(aa)
            opt = {'disp':False,'maxiter':2500}
            
            #res_cons = optimize.minimize(loss_func_lambda, lambda_0, args=(FF,delta,),jac=jac_lambda,bounds=bns,constraints=(),method='L-BFGS-B', options=opt)
            #lam = np.reshape(res_cons['x'],[TcT,1])
            #res = res_cons['fun']
            #ww = np.dot(AA.T,lam)
            
            temp1 = np.zeros([n,1])
            temp2 = np.ones([n,1])
            c = matrix(np.vstack([temp1,temp2]))
            
            temp1 = np.zeros([TcT,n])
            A_p = np.hstack([AA,temp1])
            
            temp1 = np.eye(n)
            temp2 = np.hstack([temp1,-temp1])
            temp3 = np.hstack([-temp1,-temp1])
            B = np.vstack([temp2,temp3])
            
            G = matrix(np.vstack([-A_p,B]))
            
            temp1 = delta*np.ones([TcT,1])
            temp2 = np.zeros([2*n,1])
            
            
            hh = matrix(np.vstack([-temp1,temp2]))
            aa = np.zeros([2*n,1])
            cbc = np.ones([2*n+TcT,1])
            aa = matrix(aa)
            cbc = matrix(cbc)
            primalstart = {'x':aa,'s':cbc}
            
            for iin in range(0,1):
                sol = solvers.lp(c,G,hh,primalstart=primalstart)
            
                ww = np.array(sol['x'])
                aa = np.sign(ww)
                primalstart['x'] = matrix(aa)
                
            ww2 = ww[0:n]
            W = np.zeros([n+1,1])
            W[0:ijk,0] = ww2[0:ijk,0]
            W[ijk+1:,0] = ww2[ijk:,0]
            print 'yess!'
                
            #pdb.set_trace()
        W = soft_threshold(W,sparse_thr_0)
        
        C = np.dot(A.T,A)
        C_i = np.linalg.inv(C)
        
        t_last = T0 + T_temp
        #W = np.dot(C_i,gg)
        #W = np.reshape(W,[len(W),1])
        #W = np.random.randint(-5,2,[n,1])
        #W = (W>=1).astype(int) - 5*(W<=-5).astype(int)
        #W = 0.001*W
        
        #W = np.zeros([n+1,1])
        #C_i = np.eye(n+1)
        #C = np.eye(n+1)
        u = v#-x
        #W = np.vstack([W,0])           # To account for theta
        prng = RandomState(int(time()))
        II = eta * np.eye(n+1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~Infer the Connections for Each Block~~~~~~~~~~~~~~~~~
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        yy = 0
        t_counter = 0
        ell =  int(block_size/float(t_avg))
        r_count = 0
        YY = np.zeros([ell,1]) 
        AA = np.zeros([ell,n+1])
        for t in range_T:
            
            alpha = alpha0/float(1+0.025*math.log(0.01*(t-range_T[0])+1))
            
            #........Pre-compute Some of Matrices to Speed Up the Process......
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            if (ijk in fire_t):
                t_last = t
                y = 1
            else:
                #y = 0.05
                y = -1
            
            
            if (r_count == ell):
                LAMDA = np.diag(pow(lamda,np.array(range(0,ell))))
                LAMDA_i = np.diag(pow(lamda,-np.array(range(0,ell))))
                Z = np.dot(C_i,AA.T)
                BB = LAMDA_i + pow(lamda,-ell) * np.dot(AA,Z)
                G = pow(lamda,-ell) * np.dot(Z,np.linalg.inv(BB))
                
                dw = np.dot(G,YY-np.dot(AA,W))
                s = prng.randint(0,4,[n+1,1])
                s = (s>=3).astype(int)
                #W = W - alpha * np.multiply(dw,s)
                #W = W + dw
                
                EE = C - II
                C_i = np.linalg.inv(EE)
                #W = np.dot(C_i,gg)
                
                AA = np.dot(np.diag(YY.ravel()),AA)
                TcT = len(YY)
                bc = delta*np.ones([TcT])
                AA = np.delete(AA.T,ijk,0).T
                FF = np.dot(AA,AA.T)
                
                #cons = ({'type':'ineq','fun':lambda x: np.dot(AA,x).ravel() - bc })
                
                #ww = np.reshape(np.delete(W,ijk,0).T,[n,1])
                #res_cons = optimize.minimize(loss_func, ww, jac=jac,constraints=cons,method='SLSQP', options=opt)
                #ww = np.reshape(res_cons['x'],[n,1])
                
                #if len(lam) != TcT:
                #    lambda_0 = np.zeros([TcT,1])
                #else:
                #    lambda_0 = lam
                    
                aa = np.ones([TcT,2])
                aa[:,0] = 0
                aa[:,1] = 100000
                bns = list(aa)
                #res_cons = optimize.minimize(loss_func_lambda, lambda_0, args=(FF,delta,),jac=jac_lambda,bounds=bns,constraints=(),method='L-BFGS-B', options=opt)
                #print res_cons['message']
                
                #lam = np.reshape(res_cons['x'],[TcT,1])
                
                #ww = np.dot(AA.T,lam)
                temp1 = np.zeros([n,1])
                temp2 = np.ones([n,1])
                c = matrix(np.vstack([temp1,temp2]))
                
                temp1 = np.zeros([TcT,n])
                A_p = np.hstack([AA,temp1])
                
                temp1 = np.eye(n)
                temp2 = np.hstack([temp1,-temp1])
                temp3 = np.hstack([-temp1,-temp1])
                B = np.vstack([temp2,temp3])
                
                G = matrix(np.vstack([-A_p,B]))
                
                temp1 = delta*np.ones([TcT,1])
                temp2 = np.zeros([2*n,1])
                
                
                hh = matrix(np.vstack([-temp1,temp2]))
                aa = np.zeros([2*n,1])
                cbc = np.ones([2*n+TcT,1])
                aa = matrix(aa)
                cbc = matrix(cbc)
                primalstart = {'x':matrix(ww),'s':cbc}
                
                
                cc = np.dot(AA,ww2)
                print sum(sum(cc<0))
                
                sol = solvers.lp(c,G,hh,primalstart=primalstart)
                if sol['status'] != 'optimal':
                    sol = solvers.lp(c,G,hh)
                
                ww = np.array(sol['x'])
                    
                ww2 = ww[0:n]
                
                cc = np.dot(AA,ww2)
                print sum(sum(cc<0))
                WW = np.zeros([n+1,1])
                WW[0:ijk,0] = ww2[0:ijk,0]
                WW[ijk+1:,0] = ww2[ijk:,0]
                pdb.set_trace()
                #W = W + alpha * soft_threshold(WW,sparse_thr)
                W = W + alpha * WW
                print W[0:4].T
                
                #pdb.set_trace()
                #W = 10*W/np.linalg.norm(W)
                #W = soft_threshold(W,sparse_thr)
                #C_i = pow(lamda,-ell) * (C_i - np.dot(G,Z.T))
                
                #CC = np.dot(C,C_i)
                #print np.linalg.norm(CC-np.eye(n+1))
                
                YY = np.zeros([ell,1]) 
                AA = np.zeros([ell,n+1])
                r_count = 0
            
                
            
            if (t_counter == t_avg):
                vv = vv/float(t_counter)
                xx = xx/float(t_counter)
                yy = yy/float(t_counter)
                
                u = vv-xx
                #u[-1] = 1
                #s = prng.randint(0,4,[n+1,1])
                #s = (s>=3).astype(int)
                #uu = np.multiply(u,s)
                uu = u
                
                
                AA[r_count,:] = uu.ravel()
                C = C + np.dot(uu,uu.T)
                gg = lamda * gg + theta * yy * uu   
                YY[r_count,0] = yy
                r_count = r_count + 1
                xx = np.zeros([n+1,1])
                vv = np.zeros([n+1,1])
                yy = 0
                t_counter = 0
                
                if 0:
                    C_i = lamda_i * C_i - (lamda_i * np.dot(np.dot(C_i,np.dot(uu,uu.T)),C_i))/float(lamda +np.dot(np.dot(uu.T,C_i),uu) )
                    gg = lamda * gg + yy * uu   
                    W = -np.dot(C_i,gg)
                    
                    #W = soft_threshold(W,sparse_thr)
                    
                elif 0:
                    #~~~~~~~~~~Calculate the Gradient~~~~~~~~~~~~~~
                    u[-1] = 1
                    #if np.linalg.norm(u)
                    h = eta*(np.dot(uu.T,W))
                    prng = RandomState(int(time()))
                    s = prng.randint(0,4,[n+1,1])
                    s = (s>=3).astype(int)
                    f = (math.tanh(h)-yy) * (1-pow(math.tanh(h),2)) * np.multiply(uu,s)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                    W = W - alpha * f
                    W = soft_threshold(W,sparse_thr)
                
                
                
            else:
                vv = vv + v
                xx = xx + x
                yy = yy + y
                vv[-1,0] = 1
                t_counter = t_counter + 1
            
            #................Infer the Connections for This Block..............            
            sparse_thr = sparse_thr_0/float(1+max(0,math.log((t-range_T[0]+1)/float(t_avg))))
            #b = Z - gamm *  gg
            b = gg
            
            #
            #W = W/np.linalg.norm(W)
            
            #Z = np.reshape(soft_threshold(W,sparse_thr),[n,1])
            #..................................................................    

            if t_last == t:
                x = np.zeros([n+1,1])
                v = np.zeros([n+1,1])
            
            
            #fire_t = read_spikes_lines_delayed(out_spikes_tot_mat_file,t,n,d_max,dd)
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
            v = math.exp(-1/tau_d) * v
            v[fire_t] = v[fire_t] + 1
            
            x = math.exp(-1/tau_s) * x
            x[fire_t] = x[fire_t] + 1
            
            u = v-x
            #..................................................................
            
            if (t % 1000) == 0:
                1
                
        W_inferred[:,ijk] = W
        #D_inferred[:,ijk] = D_inferred[:,ijk] + ee
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            
    return W_inferred,D_inferred





def delayed_inference_constraints_cvxopt(out_spikes_tot_mat_file,TT,n,max_itr_opt,sparse_thr_0,alpha0,theta,neuron_range):
    
    #----------------------------Initilizations--------------------------------    
    from auxiliary_functions import soft_threshold
    import os.path
    m = n
    W_inferred = np.zeros([n,m])
    
    range_tau = range(0,max_itr_opt)
    
    if len(neuron_range) == 0:
        neuron_range = np.array(range(0,m))
    
    T0 = 20000                                  # It is the offset, i.e. the time from which on we will consider the firing activity
    T_temp = 20000                              # The size of the initial batch to calculate the initial inverse matrix
    range_T = range(T_temp+T0,TT)
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV)
    delta = 0.25                                       # The tanh coefficient to approximate the sign function
    d_max = 10
    
    t_avg = 5
    block_size = 8000
    
    t0 = math.log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 2/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    #--------------------------------------------------------------------------
    
    #---------Identify Incoming Connections to Each Neuron in the List---------
    for ijk in neuron_range:
        
        print '-------------Neuron %s----------' %str(ijk)
    
    
        Z = np.reshape(W_inferred[:,ijk],[n,1])                                # The auxiliary optimization variable
    
        
        #~~~~~~~~~~~~~~~~~Calculate The Initial Inverse Matrix~~~~~~~~~~~~~~~~~
        X = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        V = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        x = np.zeros([n+1,1])
        v = np.zeros([n+1,1])
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        Y = np.zeros([1+int(T_temp/float(t_avg))])
        
        yy = 0
        
        t_counter = 0
        t_tot = 0
        for t in range(T0,T0 + T_temp):
            
            #........Pre-compute Some of Matrices to Speed Up the Process......
            #fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            if (ijk in fire_t):                
                yy = yy + 1
                x = np.zeros([n+1,1])
                v = np.zeros([n+1,1])
            
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
            x = math.exp(-1/tau_s) * x
            x[fire_t] = x[fire_t] + 1
            
            v = math.exp(-1/tau_d) * v
            v[fire_t] = v[fire_t] + 1
            
            if ((t % t_avg) == 0) and (t_counter):
                vv = vv/float(t_counter)
                xx = xx/float(t_counter)
                
                V[:,t_tot] = vv.ravel()
                X[:,t_tot] = xx.ravel()
                Y[t_tot] = yy
                
                xx = np.zeros([n+1,1])
                vv = np.zeros([n+1,1])
                yy = 0
                t_counter = 0
                t_tot = t_tot + 1
            else:
                vv = vv + v
                xx = xx + x
                vv[-1,0] = 1
                t_counter = t_counter + 1
        
        Y = np.array(Y)
        
        V = V[:,0:t_tot]
        X = X[:,0:t_tot]
        Y = Y[0:t_tot]
        
        g = (Y>0).astype(int) - (Y<=0).astype(int)
        A = (V-X).T
        #A = (V).T
        
        
        AA = np.dot(np.diag(g.ravel()),A)
        AA = np.delete(AA.T,ijk,0).T
        TcT = len(g)
        
        bc = delta*np.ones([TcT])
        
        cons = ({'type':'ineq','fun':lambda x: np.dot(AA,x).ravel() - bc })

                
        lambda_0 = np.zeros([TcT,1])
        FF = np.dot(AA,AA.T)
        
        temp1 = np.zeros([n,1])
        temp2 = np.ones([n,1])
        c = matrix(np.vstack([temp1,temp2]))
            
        temp1 = np.zeros([TcT,n])
        A_p = np.hstack([AA,temp1])
            
        temp1 = np.eye(n)
        temp2 = np.hstack([temp1,-temp1])
        temp3 = np.hstack([-temp1,-temp1])
        B = np.vstack([temp2,temp3])
            
        G = matrix(np.vstack([-A_p,B]))
            
        temp1 = delta*np.ones([TcT,1])
        temp2 = np.zeros([2*n,1])
        hh = matrix(np.vstack([-temp1,temp2]))
        
        sol = solvers.lp(c,G,hh)
            
        ww = np.array(sol['x'])
        ww2 = ww[0:n]
        W = np.zeros([n+1,1])
        W[0:ijk,0] = ww2[0:ijk,0]
        W[ijk+1:,0] = ww2[ijk:,0]
        x = np.zeros([n,1])
        v = np.zeros([n,1])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        
        #~~~~~~~~~~~~~~~~~Infer the Connections for Each Block~~~~~~~~~~~~~~~~~
        for ttau in range_tau:
            alpha = alpha0/float(1+math.log(ttau+1))
            sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
            
            #~~~~~~~~~~~~~~~~~~~~~~~~Initialize the Values~~~~~~~~~~~~~~~~~~~~~~~~~
            t_last = T0 + T_temp
            
            xx = np.zeros([n,1])
            vv = np.zeros([n,1])
            #vv[:-1] = 1
            u = vv-xx
            yy = 0
            t_counter = 0
            ell =  int(block_size/float(t_avg))
            r_count = 0
            YY = np.zeros([ell,1]) 
            AA = np.zeros([ell,n+1])
            
            prng = RandomState(int(time()))
            #t_avg = 1
            #II = eta * np.eye(n+1)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
            if ttau > 0:
                range_T = range(T0,TT)
                t_last = T0 
            
            for t in range_T:
                #........Pre-compute Some of Matrices to Speed Up the Process......
                fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
                if (ijk in fire_t):
                    t_last = t
                    y = 1
                else:
                    y = -1
            
                
                if (t_counter == t_avg):
                    vv = vv/float(t_counter)
                    xx = xx/float(t_counter)
                    yy = yy/float(t_counter)
                    
                    uu = vv-xx
                    #u[-1] = 1
                    
                    TcT = 1
                    
                    AA = np.reshape(uu,[TcT,n])
                    YY = yy * np.ones([1,1])
                    
                    AA = np.dot(np.diag(YY.ravel()),AA)
                    cc = np.dot(AA,ww2)
                    if sum(cc < delta):
                        #AA = np.delete(AA.T,ijk,0).T
                        
                        temp1 = np.zeros([n,1])
                        temp2 = np.ones([n,1])
                        cc = matrix(np.vstack([temp1,temp2]))
                        
                        temp1 = np.zeros([TcT,n])
                        A_p = np.hstack([AA,temp1])
                        
                        temp1 = np.eye(n)
                        temp2 = np.hstack([temp1,-temp1])
                        temp3 = np.hstack([-temp1,-temp1])
                        B = np.vstack([temp2,temp3])
                        
                        G = matrix(np.vstack([-A_p,B]))
                        
                        temp1 = delta*np.ones([TcT,1])
                        temp2 = np.zeros([2*n,1])
                        
                        hh = matrix(np.vstack([-temp1,temp2]))
                        
                        sol = solvers.lp(cc,G,hh)
                        
                        #------Print Costs------
                        #cc = np.dot(AA,ww2)
                        #print sum(sum(cc<0))
                        #------------------------
                        if 'optimal' in sol['status']:
                            ww = np.array(sol['x'])
                            ww2 = ww[0:n]
                            
                            WW = np.zeros([n+1,1])
                            WW[0:ijk,0] = ww2[0:ijk,0]
                            WW[ijk+1:,0] = ww2[ijk:,0]
                            
                            W = W + alpha * WW
                            W = soft_threshold(W,sparse_thr)
                        else:
                            print 'dman %s' %str(t)
                    
                    xx = np.zeros([n,1])
                    vv = np.zeros([n,1])
                    yy = 0
                    t_counter = 0
                    
                else:
                    vv = vv + v
                    xx = xx + x
                    yy = yy + y
                    #vv[-1,0] = 1
                    t_counter = t_counter + 1
                
    
                if t_last == t:
                    x = np.zeros([n,1])
                    v = np.zeros([n,1])
                
                
                #fire_t = read_spikes_lines_delayed(out_spikes_tot_mat_file,t,n,d_max,dd)
                fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
                v = math.exp(-1/tau_d) * v
                v[fire_t] = v[fire_t] + 1
                
                x = math.exp(-1/tau_s) * x
                x[fire_t] = x[fire_t] + 1
                
                u = v-x
                
                #if (t % 50002) == 0:
                #    pdb.set_trace()
                #..................................................................
               
            pdb.set_trace()
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            W_inferred[:,ijk] = W
            
    return W_inferred



#------------------------------------------------------------------------------
def delayed_inference_constraints_numpy(out_spikes_tot_mat_file,TT,n,max_itr_opt,sparse_thr_0,alpha0,theta,neuron_range):
    
    #----------------------------Initilizations--------------------------------    
    from auxiliary_functions import soft_threshold
    import os.path
    m = n
    W_inferred = np.zeros([n,m])
    
    range_tau = range(0,max_itr_opt)
    
    if len(neuron_range) == 0:
        neuron_range = np.array(range(0,m))
    
    T0 = 20000                                  # It is the offset, i.e. the time from which on we will consider the firing activity
    T_temp = 20000                              # The size of the initial batch to calculate the initial inverse matrix
    range_T = range(T_temp+T0,TT)
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV)
    delta = 0.25                                       # The tanh coefficient to approximate the sign function
    d_max = 10
    t_gap = 25                                     # The gap between samples to consider
    t_avg = 2
    block_size = 8000
    
    t0 = math.log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 2/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    #--------------------------------------------------------------------------
    
    #---------Identify Incoming Connections to Each Neuron in the List---------
    for ijk in neuron_range:
        
        print '-------------Neuron %s----------' %str(ijk)
    
    
        Z = np.reshape(W_inferred[:,ijk],[n,1])                                # The auxiliary optimization variable
        Z = Z[1:,0]
    
        
        #~~~~~~~~~~~~~~~~~Calculate The Initial Inverse Matrix~~~~~~~~~~~~~~~~~
        X = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        V = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        x = np.zeros([n+1,1])
        v = np.zeros([n+1,1])
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        Y = np.zeros([1+int(T_temp/float(t_avg))])
        
        yy = 0
        
        t_counter = 0
        t_tot = 0
        for t in range(T0,T0 + T_temp):
            
            #........Pre-compute Some of Matrices to Speed Up the Process......
            #fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            if (ijk in fire_t):                
                yy = yy + 1
                x = np.zeros([n+1,1])
                v = np.zeros([n+1,1])
            
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
            x = math.exp(-1/tau_s) * x
            x[fire_t] = x[fire_t] + 1
            
            v = math.exp(-1/tau_d) * v
            v[fire_t] = v[fire_t] + 1
            
            if ((t % t_avg) == 0) and (t_counter):
                vv = vv/float(t_counter)
                xx = xx/float(t_counter)
                
                V[:,t_tot] = vv.ravel()
                X[:,t_tot] = xx.ravel()
                Y[t_tot] = yy
                
                xx = np.zeros([n+1,1])
                vv = np.zeros([n+1,1])
                yy = 0
                t_counter = 0
                t_tot = t_tot + 1
            else:
                vv = vv + v
                xx = xx + x
                vv[-1,0] = 1
                t_counter = t_counter + 1
        
        Y = np.array(Y)
        
        V = V[:,0:t_tot]
        X = X[:,0:t_tot]
        Y = Y[0:t_tot]
        
        
        g_der = np.nonzero(Y)[0]
        
        g = (Y>0).astype(int) - (Y<=0).astype(int)
        A = (V-X).T
        #A = (V).T
        
        AA = np.dot(np.diag(g.ravel()),A)
        
        #--------------Go For Derviative Maximization--------------
        B = A[g_der,:]-A[g_der-1,:]
        #----------------------------------------------------------
        
        AA = np.vstack([AA,B])
        AA = np.delete(AA.T,ijk,0).T
        TcT = AA.shape[0]#len(g)
        
        bc = delta*np.ones([TcT])
        
        cons = ({'type':'ineq','fun':lambda x: np.dot(AA,x).ravel() - bc })

                
        lambda_0 = np.zeros([TcT,1])
        FF = np.dot(AA,AA.T)
        
        cons = ({'type':'ineq','fun':lambda x: np.dot(AA,x).ravel() - bc })

        #opt = {'disp':False,'maxiter':500}
        
        
        lambda_0 = np.zeros([TcT,1])
        FF = np.dot(AA,AA.T)
        Z = np.zeros([n,1])
        aa = np.ones([TcT,2])
        aa[:,0] = 0
        aa[:,1] = 100000
        bns = list(aa)
        opt = {'disp':False,'maxiter':2500}
            
        for i in range(0,2):
            res_cons = optimize.minimize(loss_func_lambda, lambda_0, args=(FF,delta,AA,Z),jac=jac_lambda,bounds=bns,constraints=(),method='L-BFGS-B', options=opt)
            lam = np.reshape(res_cons['x'],[TcT,1])
            ww = np.dot(AA.T,lam)
            ww2 = Z + 0.5*ww[0:n]
            Z = soft_threshold(ww2,sparse_thr_0)
            Z = Z/np.linalg.norm(Z)
        
        #cc = np.dot(AA,Z)
        #print sum(cc>0)
        
        W = np.zeros([n+1,1])
        W[0:ijk,0] = Z[0:ijk,0]
        W[ijk+1:,0] = Z[ijk:,0]
        
        t_last = T0 + T_temp
        
        u = v-x
        
        prng = RandomState(int(time()))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~Infer the Connections for Each Block~~~~~~~~~~~~~~~~~
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        yy = 0
        t_counter = 0
        ell =  int(block_size/float(t_avg))
        r_count = 0
        YY = np.zeros([ell,1]) 
        AA = np.zeros([ell,n+1])
        
        for ttau in range(0,500):
            alpha = alpha0/float(1+math.log(ttau+1))
            sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
            
            for t in range_T:
                
                #........Pre-compute Some of Matrices to Speed Up the Process......
                fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
                if (ijk in fire_t):
                    t_last = t
                    y = 1
                else:
                    y = -1
                
                
                if (r_count == ell):
                    s = prng.randint(0,4,[n+1,1])
                    s = (s>=3).astype(int)

                    t_init = np.random.randint(0,t_gap)
                    t_inds = np.array(range(t_init,ell,t_gap))
                    
                    Y = YY + np.ones(YY.shape)
                    
                    g_der = np.nonzero(Y)[0]
                    
                    #--------------Go For Derviative Maximization--------------
                    B = AA[g_der,:]-AA[g_der-1,:]
                    #----------------------------------------------------------
                    
                    AA = AA[t_inds,:]
                    YY = YY[t_inds,0]
                    AA = np.dot(np.diag(YY.ravel()),AA)
                    AA = np.vstack([AA,B])
                    AA = np.delete(AA.T,ijk,0).T
                    
                    TcT = AA.shape[0]
                    
                    bc = delta*np.ones([TcT])
                    FF = np.dot(AA,AA.T)
                    
                    aa = np.ones([TcT,2])
                    aa[:,0] = 0
                    aa[:,1] = 100000
                    bns = list(aa)
                    lambda_0 = np.zeros([TcT,1])
                    for i in range(0,10):
                        res_cons = optimize.minimize(loss_func_lambda, lambda_0, args=(FF,delta,AA,Z),jac=jac_lambda,bounds=bns,constraints=(),method='L-BFGS-B', options=opt)
                        lam = np.reshape(res_cons['x'],[TcT,1])
                        ww = np.dot(AA.T,lam)
                        ww2 = Z + 0.5*ww[0:n]
                        Z = soft_threshold(ww2,sparse_thr)
                        Z = Z/np.linalg.norm(Z)
                        if sum(Z) == 0:
                            pdb.set_trace()
                
                    WW = np.zeros([n+1,1])
                    WW[0:ijk,0] = Z[0:ijk,0]
                    WW[ijk+1:,0] = Z[ijk:,0]
                    
                    #W = W + alpha * soft_threshold(WW,sparse_thr)
                    W = W - alpha * WW
                    W = W/np.linalg.norm(W)
                    cc = np.dot(AA,Z)
                    print sum(sum(cc<0))
                    
                    YY = np.zeros([ell,1]) 
                    AA = np.zeros([ell,n+1])
                    r_count = 0
                
                    
                
                if (t_counter == t_avg):
                    vv = vv/float(t_counter)
                    xx = xx/float(t_counter)
                    yy = yy/float(t_counter)
                    
                    u = vv-xx
                    #u[-1] = 1
                    #s = prng.randint(0,4,[n+1,1])
                    #s = (s>=3).astype(int)
                    #uu = np.multiply(u,s)
                    uu = u
                    
                    AA[r_count,:] = uu.ravel()
                    YY[r_count,0] = yy
                    r_count = r_count + 1
                    xx = np.zeros([n+1,1])
                    vv = np.zeros([n+1,1])
                    yy = 0
                    t_counter = 0
                    
                else:
                    vv = vv + v
                    xx = xx + x
                    yy = yy + y
                    #vv[-1,0] = 1
                    t_counter = t_counter + 1
                
                if t_last == t:
                    x = np.zeros([n+1,1])
                    v = np.zeros([n+1,1])
                
                
                #fire_t = read_spikes_lines_delayed(out_spikes_tot_mat_file,t,n,d_max,dd)
                fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
                v = math.exp(-1/tau_d) * v
                v[fire_t] = v[fire_t] + 1
                
                x = math.exp(-1/tau_s) * x
                x[fire_t] = x[fire_t] + 1
                
                u = v-x
                #..................................................................
            
            
            Z = (Z>2*sparse_thr).astype(int) - (Z<-2*sparse_thr).astype(int)   
            #pdb.set_trace()
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Predict Spikes~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        X = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        V = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        x = np.zeros([n+1,1])
        v = np.zeros([n+1,1])
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        Y = np.zeros([1+int(T_temp/float(t_avg))])
        
        yy = 0
        
        t_counter = 0
        t_tot = 0
        for t in range(T0,T0 + T_temp):
            
            #........Pre-compute Some of Matrices to Speed Up the Process......
            #fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            if (ijk in fire_t):                
                yy = yy + 1
                x = np.zeros([n+1,1])
                v = np.zeros([n+1,1])
            
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
            x = math.exp(-1/tau_s) * x
            x[fire_t] = x[fire_t] + 1
            
            v = math.exp(-1/tau_d) * v
            v[fire_t] = v[fire_t] + 1
            
            if ((t % t_avg) == 0) and (t_counter):
                vv = vv/float(t_counter)
                xx = xx/float(t_counter)
                
                V[:,t_tot] = vv.ravel()
                X[:,t_tot] = xx.ravel()
                Y[t_tot] = yy
                
                xx = np.zeros([n+1,1])
                vv = np.zeros([n+1,1])
                yy = 0
                t_counter = 0
                t_tot = t_tot + 1
            else:
                vv = vv + v
                xx = xx + x
                vv[-1,0] = 1
                t_counter = t_counter + 1
        
        Y = np.array(Y)
        
        V = V[:,0:t_tot]
        X = X[:,0:t_tot]
        Y = Y[0:t_tot]
        
        g = (Y>0).astype(int) - (Y<=0).astype(int)
        A = (V-X).T
        
        Y_predict = np.dot(A,W)
        pdb.set_trace()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        W_inferred[:,ijk] = W
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            
    return W_inferred



#------------------------------------------------------------------------------
def delayed_inference_constraints_svm(out_spikes_tot_mat_file,TT,n,max_itr_opt,sparse_thr_0,alpha0,theta,neuron_range):
    
    #---------------------------------------------------------------------------
    from sklearn import svm
    from sklearn.ensemble import AdaBoostClassifier
    
    #clf = svm.LinearSVC()
    clf = AdaBoostClassifier()
    clf.base_estimator=svm.SVC(kernel = 'linear')
    clf.algorithm = 'SAMME'
    clf.kernel = 'linear'
    
    #clf.C = 5000
    #clf.tol = 1e-3
    #clf.decision_function_shape = 'ovo'
    #clf.multi_class = 'crammer_singer'
    #clf.kernel = 'rbf'
    #clf.degree = 3
    #clf.random_state = RandomState(int(time()))
    #---------------------------------------------------------------------------
    
    #----------------------------Initilizations--------------------------------    
    from auxiliary_functions import soft_threshold
    import os.path
    m = n
    W_inferred = np.zeros([n,m])
    
    range_tau = range(0,max_itr_opt)
    
    if len(neuron_range) == 0:
        neuron_range = np.array(range(0,m))
    
    if TT > 40000:
        T0 = 20000                                  # It is the offset, i.e. the time from which on we will consider the firing activity
        T_temp = 4000                              # The size of the initial batch to calculate the initial inverse matrix
        block_size = 10000
    else:
        T0 = 0                                  # It is the offset, i.e. the time from which on we will consider the firing activity
        T_temp = 500                              # The size of the initial batch to calculate the initial inverse matrix
        block_size = 500
    range_T = range(T_temp+T0,TT)
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV)
    delta = 0.25                                       # The tanh coefficient to approximate the sign function
    d_max = 10
    t_gap = 3                                     # The gap between samples to consider
    t_avg = 1
    
    
    W_infer = np.zeros([int(len(range_T)/float(block_size))+1,n+1])
    
    t0 = math.log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 2/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    #--------------------------------------------------------------------------
    
    #---------Identify Incoming Connections to Each Neuron in the List---------
    for ijk in neuron_range:
        
        print '-------------Neuron %s----------' %str(ijk)
    
    
        Z = np.reshape(W_inferred[:,ijk],[n,1])                                # The auxiliary optimization variable
        Z = Z[1:,0]
    
        
        #~~~~~~~~~~~~~~~~~Calculate The Initial Inverse Matrix~~~~~~~~~~~~~~~~~
        X = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        V = np.zeros([n+1,1+int(T_temp/float(t_avg))])
        x = np.zeros([n+1,1])
        v = np.zeros([n+1,1])
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        Y = np.zeros([1+int(T_temp/float(t_avg))])
        
        yy = 0
        
        t_counter = 0
        t_tot = 0
        for t in range(T0,T0 + T_temp):
            
            #........Pre-compute Some of Matrices to Speed Up the Process......
            #fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            if (ijk in fire_t):                
                yy = yy + 1
                x = np.zeros([n+1,1])
                v = np.zeros([n+1,1])
            
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
            x = math.exp(-1/tau_s) * x
            x[fire_t] = x[fire_t] + 1
            
            v = math.exp(-1/tau_d) * v
            v[fire_t] = v[fire_t] + 1
            
            
            V[:,t_tot] = v.ravel()
            X[:,t_tot] = x.ravel()
            Y[t_tot] = yy
            t_tot = t_tot + 1
            yy = 0
        
        Y = np.array(Y)
        
        V = V[:,0:t_tot]
        X = X[:,0:t_tot]
        Y = Y[0:t_tot]
        
        
        g_der = np.nonzero(Y)[0]
        
        g = (Y>0).astype(int) - (Y<=0).astype(int)
        A = (V-X).T
        #A = (V).T
        A = (A>0.85).astype(int)
        
        ell = int(T_temp/float(t_avg))
        t_init = np.random.randint(0,t_gap)
        t_inds = np.array(range(t_init,ell,t_gap))
        
        #A = A[t_inds,:]
        #g = g[t_inds]
        A = np.delete(A.T,ijk,0).T
        
        #------------------Train the Classifier--------------------
        features_projected_train = A
        actual_vals_train = g
        
        W = np.zeros([n+1,1])
        clf.sample_weight = {1:50,-1:1}
        if (1 in actual_vals_train) and (-1 in actual_vals_train):
            clf.fit(features_projected_train, actual_vals_train.ravel())
        
        
        #sc = clf.score(features_projected_train, actual_vals_train.ravel(), sample_weight=None)
        
            sc = clf.score(features_projected_train, actual_vals_train.ravel())
            print sc
            est = clf.estimators_
            est_w = clf.estimator_weights_
            
            ww = np.zeros([n,1])
            for ili in range(0,len(est)):
                aa = est[ili];bb = aa.coef_;ww = ww + est_w[ili] * bb.T
            
            
            W[0:ijk,0] = ww[0:ijk,0]
            W[ijk+1:,0] = ww[ijk:,0]
            
        else:
            print "Oneof the classes is empty. moving on!"
        #----------------------------------------------------------
        
        
        
        
        t_last = T0 + T_temp
        
        u = v-x
        
        
        prng = RandomState(int(time()))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~Infer the Connections for Each Block~~~~~~~~~~~~~~~~~
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        yy = 0
        t_counter = 0
        ell =  int(block_size/float(t_avg))
        r_count = 0
        YY = np.zeros([ell,1]) 
        AA = np.zeros([ell,n+1])
        W2 = np.zeros([n+1,1])
        
        for ttau in range(0,500):
            alpha = alpha0/float(1+math.log(ttau+1))
            sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
            itr_W = 0
            
            for t in range_T:
                
                #........Pre-compute Some of Matrices to Speed Up the Process......
                fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
                if (ijk in fire_t):
                    t_last = t
                    y = 1
                else:
                    y = -1
                
                
                if (r_count == ell):
                    s = prng.randint(0,4,[n+1,1])
                    s = (s>=3).astype(int)

                    t_init = np.random.randint(0,t_gap)
                    t_inds = np.array(range(t_init,ell,t_gap))
                    
                    Y = YY + np.ones(YY.shape)
                    Y = (Y>0).astype(int) - (Y<=0).astype(int)
                    g_der = np.nonzero(Y)[0]
                    
                    
                    AA = AA[t_inds,:]
                    Y = Y[t_inds,0]
                    AA = np.delete(AA.T,ijk,0).T
                    AA = (AA>0.85).astype(int)
                    
                    features_projected_train = AA
                    actual_vals_train = Y
                    
                    if (1 in Y) and (-1 in Y):
                        aa = np.nonzero(actual_vals_train>0)[0]
                        ll = 2*len(aa)
                        bb = np.random.randint(0,len(actual_vals_train),ll)
                        temp1 = np.reshape(actual_vals_train[aa],[len(actual_vals_train[aa]),1])
                        
                        actual_vals_train = -1*np.ones([len(aa)+len(bb)])
                        actual_vals_train[0:len(aa)] = 1
                        
                        features_projected_train = np.zeros([len(aa)+len(bb),n])
                        features_projected_train[0:len(aa),:] = AA[aa,:]
                        features_projected_train[len(aa):len(aa)+len(bb),:] = AA[bb,:]
                        clf.fit(features_projected_train, actual_vals_train.ravel())
                        
                        sc = clf.score(features_projected_train, actual_vals_train.ravel())
                        
                        
                        
                        #----------------------------------------------------------
            
                        est = clf.estimators_
                        est_w = clf.estimator_weights_
                        ww = np.zeros([n,1])
                        if sc > (1.02*len(bb)/len(actual_vals_train)):
                            print sc
                            for ili in range(0,len(est)):
                                aa = est[ili];bb = aa.coef_;ww = ww + est_w[ili] * bb.T
                            
                            WW = np.zeros([n+1,1])
                            WW[0:ijk,0] = ww[0:ijk,0]
                            WW[ijk+1:,0] = ww[ijk:,0]
                            
                            W = W + sc * WW
                            #W = W/np.linalg.norm(W)
                            W = W/(np.abs(W)).max()
                            W_infer[itr_W,:] = sc * WW.ravel()
                            itr_W = itr_W + 1
                            
                            
                            
                            #W = W + alpha * soft_threshold(WW,sparse_thr)
                            #W = W - alpha * soft_threshold(WW,sparse_thr)
                            #W = W/np.linalg.norm(W)
                        else:
                            print 'Score was unsuccesfull: %f ' %sc
                    else:
                        print 'Something is fishy!'
                    
                    YY = np.zeros([ell,1]) 
                    AA = np.zeros([ell,n+1])
                    r_count = 0
                
                    
                
                if (t_counter == t_avg):
                    vv = vv/float(t_counter)
                    xx = xx/float(t_counter)
                    yy = yy/float(t_counter)
                    
                    u = vv-xx
                    #u[-1] = 1
                    #s = prng.randint(0,4,[n+1,1])
                    #s = (s>=3).astype(int)
                    #uu = np.multiply(u,s)
                    uu = u
                    
                    AA[r_count,:] = uu.ravel()
                    YY[r_count,0] = yy
                    r_count = r_count + 1
                    xx = np.zeros([n+1,1])
                    vv = np.zeros([n+1,1])
                    yy = 0
                    t_counter = 0
                    
                else:
                    vv = vv + v
                    xx = xx + x
                    yy = yy + y
                    #vv[-1,0] = 1
                    t_counter = t_counter + 1
                
                if t_last == t:
                    x = np.zeros([n+1,1])
                    v = np.zeros([n+1,1])
                
                
                #fire_t = read_spikes_lines_delayed(out_spikes_tot_mat_file,t,n,d_max,dd)
                fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
                v = math.exp(-1/tau_d) * v
                v[fire_t] = v[fire_t] + 1
                
                x = math.exp(-1/tau_s) * x
                x[fire_t] = x[fire_t] + 1
                
                u = v-x
                #..................................................................
            
            if itr_W > 1:
                W2 = W2 + np.reshape(W_infer[0:itr_W,:].mean(axis = 0),[n+1,1])
            elif itr_W == 1:
                W2 = W2 + np.reshape(W_infer[0:itr_W,:],[n+1,1])
                
            #if ((ttau + 1) %
            #pdb.set_trace()
            #Z = (Z>4*sparse_thr).astype(int) - (Z<-4*sparse_thr).astype(int)   
            
        #pdb.set_trace()    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Predict Spikes~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if 0:
            X = np.zeros([n+1,1+int(T_temp/float(t_avg))])
            V = np.zeros([n+1,1+int(T_temp/float(t_avg))])
            x = np.zeros([n+1,1])
            v = np.zeros([n+1,1])
            xx = np.zeros([n+1,1])
            vv = np.zeros([n+1,1])
            Y = np.zeros([1+int(T_temp/float(t_avg))])
            
            yy = 0
            
            t_counter = 0
            t_tot = 0
            for t in range(T0,T0 + T_temp):
                
                #........Pre-compute Some of Matrices to Speed Up the Process......
                #fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
                fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
                if (ijk in fire_t):                
                    yy = yy + 1
                    x = np.zeros([n+1,1])
                    v = np.zeros([n+1,1])
                
                fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
                x = math.exp(-1/tau_s) * x
                x[fire_t] = x[fire_t] + 1
                
                v = math.exp(-1/tau_d) * v
                v[fire_t] = v[fire_t] + 1
                
                if ((t % t_avg) == 0) and (t_counter):
                    vv = vv/float(t_counter)
                    xx = xx/float(t_counter)
                    
                    V[:,t_tot] = vv.ravel()
                    X[:,t_tot] = xx.ravel()
                    Y[t_tot] = yy
                    
                    xx = np.zeros([n+1,1])
                    vv = np.zeros([n+1,1])
                    yy = 0
                    t_counter = 0
                    t_tot = t_tot + 1
                else:
                    vv = vv + v
                    xx = xx + x
                    vv[-1,0] = 1
                    t_counter = t_counter + 1
            
            Y = np.array(Y)
            
            V = V[:,0:t_tot]
            X = X[:,0:t_tot]
            Y = Y[0:t_tot]
            
            g = (Y>0).astype(int) - (Y<=0).astype(int)
            A = (V-X).T
            
            Y_predict = np.dot(A,W)
            pdb.set_trace()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        
        W_inferred[0:n,ijk] = W2[0:n].ravel()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            
    return W_inferred
