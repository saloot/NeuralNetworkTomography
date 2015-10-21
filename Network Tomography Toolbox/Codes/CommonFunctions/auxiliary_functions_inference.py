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
#from scipy.optimize import minimize,linprog
from cvxopt import solvers, matrix, spdiag, log

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
    
    n,TT = spikes_times.shape
    m = n
    W_inferred = np.zeros([n,m])
    D_inferred = np.zeros([n,m])
    
    range_tau = range(0,max_itr_opt)
    
    if len(neuron_range) == 0:
        neuron_range = np.array(range(0,m))
    
    gamm = 10                                       # Determine how much sparsity is important for the algorithm (the higher the more important)
    
    dl = 0#
    
    T_temp = TT-1 #min(2000,TT-1) #TT-1#            # Decide if the algorithm should divide the spike times into several smaller blocks and merge the results
    range_T = range(T_temp,TT,T_temp)
    W_total = {}
    D_total = {}
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV)
    t0 = log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 1/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    #--------------------------------------------------------------------------
    
    
    #---------------Preprocess Spike Times and the Integration Effect----------
    #CC = np.roll(spikes_times,1, axis=1)           # Shift the spikes time one ms to account for causality and minimum propagation delay
    CC = spikes_times
    
    V = np.zeros([n,TT])
    X = np.zeros([n,TT])
    
    AA = np.reshape(np.array(range(1,TT+1)),[TT,1])
    AA = np.dot(AA,np.ones([1,n]))
    AA = np.multiply(CC,AA.T)
    
    
    for jj in range(dl,TT-1):
        DD = AA[:,max(0,jj-d_window):jj]
        
        V[:,jj] = np.sum(np.multiply(np.sign(DD),np.exp(-(jj-DD-dl)/tau_d)),axis = 1)
        X[:,jj] = np.sum(np.multiply(np.sign(DD),np.exp(-(jj-DD-dl)/tau_s)),axis = 1)
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
        if len(D_act):
            dd = D_act[:,ijk]
            CC = np.zeros([n,TT])
            for i in range(0,n):
                CC[i,:] = np.roll(spikes_times[i,:],int(dd[i]*1000)+1)
                CC[i,0:int(dd[i]*1000)+1] = 0
            
            AA = np.reshape(np.array(range(1,TT+1)),[TT,1])
            AA = np.dot(AA,np.ones([1,n]))
            AA = np.multiply(CC,AA.T)
            d_window = 0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Experimental Block~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~Extent the Integration Window to the Last Reset~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if 0:
            for jj in range(dl,TT-1):
                if jj in t_fire:
                    t_last = jj
                t_min = max(0,t_last-d_window)
                R[:,jj] = np.sum(CC[:,t_min:jj-dl],axis = 1)
                DD = AA[:,t_min:jj-dl]
                
                V[:,jj] = np.sum(np.multiply(np.sign(DD),np.exp(-(jj-DD-dl)/tau_d)),axis = 1)
                X[:,jj] = np.sum(np.multiply(np.sign(DD),np.exp(-(jj-DD-dl)/tau_s)),axis = 1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Experimental Block~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~Some Tests to Verify the Algorithm When Connectivity Is Known~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if len(W_act) and len(D_act):            
            dd = np.reshape(dd,[n,1])
            dd1 = np.multiply(np.sign(dd),np.exp(dd/(0.001*tau_d)))
            dd2 = np.multiply(np.sign(dd),np.exp(dd/(0.001*tau_s)))
            
            DD1 = np.diag(dd1[:,0])
            DD2 = np.diag(dd2[:,0])
            w = W_act[:,ijk]
            w = np.reshape(w,[n,1])
            
            U = np.dot(V.T,DD1) - np.dot(X.T,DD2)
            
            U = np.multiply(U,(U>0).astype(int))
            U = np.multiply(U,(U<1/U0).astype(int)) + np.multiply(1/U0,(U>1/U0).astype(int))
            
            U = V.T-X.T
            
            H = U0*np.dot(U,w)
            
            hh = (H>theta*.001).astype(int)
            pdb.set_trace()
            #plt.plot(hh[0:100]);plt.plot(Y[0:100],'r');plt.show()
            #plt.plot(H[0:300]);plt.plot(0.1*Y[0:300],'r');plt.show()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        
        #~~~~~~~~~~~~~~~~~Infer the Connections for Each Block~~~~~~~~~~~~~~~~~
        for T_T in range_T:
            
            #...................In-loop Initializations........................
            Y = spikes_times[ijk,T_T-T_temp:T_T]                    # Output firing pattern
            
            DD1 = exp(1.5/tau_d)*np.eye(n)                          # The first diagona delay matrix (corresponding to the decay coefficient)
            DD2 = exp(1.5/tau_s)*np.eye(n)                          # The second diagona delay matrix (corresponding to the rise coefficient)
            
            WW = W_inferred[:,ijk]
            Z = np.reshape(WW,[n,1])                                # The auxiliary optimization variable 
            #..................................................................
            
            #..........Construct the Positive and Zero State Matrices..........            
            bb = np.nonzero(Y)
            
            VV = V[:,T_T-T_temp:T_T]
            XX = X[:,T_T-T_temp:T_T]
            
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
            B_i = np.linalg.pinv(B)
            g = ((theta-h0)*np.vstack([np.ones([T1,1]),-np.ones([T2,1])]) + 5.05*np.ones([T1+T2,1]))/U0
            U_i =  np.linalg.pinv(U)
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
                        for ttau in range(0,range_tau):
                            
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
                    C = gamm * np.eye(n) - np.dot(U.T,U)
                    C_i = np.linalg.inv(C)
                        
                    for ttau in range_tau:
                        
                        #~~~~~~~~~~~~~In-Loop Initializations~~~~~~~~~~~~                    
                        sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
                        b = Z - gamm *  np.dot(U.T,g)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                
                        #~~~~~~~~~~~Minimize with Respect to W~~~~~~~~~~~
                        W = np.dot(C_i,b)
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
                    
                    if len(W_act):
                        W = 1000*(np.reshape(W_act[:,ijk],[n,1]))
                        U_i = np.linalg.pinv(np.dot(XX-U,np.diag(W[:,0])))
                        #U_i = np.linalg.pinv(np.dot(U,np.diag(W[:,0])))
                        
                        aa = np.dot(np.dot(U-XX,np.diag(W[:,0])),np.ones([n,1]))
                        
                        dd = np.dot(U_i,g+1*aa)
                            
                        #for i in range(0,n):
                        #    if abs(W[i]) > 0:
                        #        dd[i] =dd[i]/W[i]
                        #    else:
                        #        dd[i] = 0
                        #dd = np.multiply(dd,pow(W+0.00001*np.random.rand(n,1),-1))
                        #dd = np.multiply(dd,(dd>0).astype(int))
                        #pdb.set_trace()
                        #D_inferred[:,ijk] = dd[:,0]
                        if (T_T == range_T[0]):
                            D_total[str(ijk)] = [dd]
                        else:
                            D_total[str(ijk)].append(dd)
                    else:
                        dd = D_act[:,ijk]
                        dd1 = np.multiply(np.sign(dd),np.exp(dd/(0.001*tau_d)))
                        dd2 = np.multiply(np.sign(dd),np.exp(dd/(0.001*tau_s)))
                        DD1 = np.diag(dd1)
                        DD2 = np.diag(dd2)
                        A = (np.dot(U,DD1)-np.dot(XX,DD2))#np.dot(C,np.dot(UU,DD))
                        gamm = 0.1
                        C = gamm * np.eye(n) - np.dot(A.T,A)
                        C_i = np.linalg.inv(C)
                        for ttau in range(0,10):#range_tau:
                        
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
                        
                        if (T_T == range_T[0]):
                            W_total[str(ijk)] = [W]
                        else:
                            W_total[str(ijk)].append(W)
                    
            #..................................................................
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~Merge the Results of Different Blocks~~~~~~~~~~~~~~~~~
        if len(W_act)  == 0:           
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
            W_inferred[:,ijk] = W_inferred[:,ijk] + ee 

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
