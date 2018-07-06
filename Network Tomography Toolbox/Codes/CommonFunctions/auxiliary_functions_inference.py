#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
#from brian import *
from scipy import sparse,linalg
import warnings
import pdb,os,sys
import random
import copy
import numpy.ma as ma
try:
    import matplotlib.pyplot as plt
except:
    pass
import numpy as np
import math
from default_values import *
import time
import resource
from scipy.cluster.vq import kmeans,whiten,kmeans2,vq
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
            if opt == '-T':
                no_stimul_rounds = int(arg)                         # Number of rcorded samples 
            elif opt == '-A':
                file_name_data = str(arg)                           # The file that contains the data
            elif opt == '-Q':
                no_processes = int(arg)                             # The number of cpu cores to use for simulaitons
            elif opt == '-N':
                no_neurons = int(arg)                               # Number of observed eurons
            elif opt == '-M':
                inference_method = int(arg)                         # The inference method
            elif opt == '-H':
                no_hidden_neurons = int(arg)                         # The number of neurons to artificially hide
            elif opt == '-Y':
                sparsity_flag = float(arg)                            # The flag that determines if sparsity should be observed during inference
            elif opt == '-X':
                infer_itr_max = int(arg)                            # The flag that determines if sparsity should be observed during inference            
            elif opt == '-U': 
                beta = int(arg)                                     # Specify the update probability paramter (p = 1/beta) in STOCHASTIC NEUINF
            elif opt == '-J': 
                class_sample_freq = float(arg)                      # Specify the probability of choosing samples from the firing instances
            elif opt == '-L': 
                kernel_choice = str(arg)                      # Specify the integration kernel
            elif opt == '-Z': 
                alpha0 = float(arg)                                 # Specify the update learnining rate
            elif opt == '-b': 
                bin_size = int(arg)                                 # If it is nonzero, the spikes will be placed within bins of size "bin_size"
            elif opt == '-S': 
                block_size = int(arg)                               # The size of the smaller blocks to divide the spike file into
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
    if 'infer_itr_max' not in locals():
        infer_itr_max = INFERENCE_ITR_MAX_DEFAULT
        print('ATTENTION: The default value of %s for infer_itr_max is considered.\n' %str(infer_itr_max))
        
    if 'no_stimul_rounds' not in locals():        
        no_stimul_rounds = 0#NO_STIMUL_ROUNDS_DEFAULT
        #print('ATTENTION: The default value of %s for no_stimul_rounds is considered.\n' %str(no_stimul_rounds))

    if 'file_name_data' not in locals():
        file_name_data = ''
        
    if 'no_processes' not in locals():
        no_processes = NO_CPUS_DEFAULT
        print('ATTENTION: The default value of %s for no_processes is considered.\n' %str(NO_CPUS_DEFAULT))

    if 'ternary_mode' not in locals():
        ternary_mode = TERNARY_MODE_DEFAULT;
        print('ATTENTION: The default value of %s for ternary_mode is considered.\n' %str(ternary_mode))

    if 'file_name_base_results' not in locals():
        file_name_base_results = FILE_NAME_BASE_RESULT_DEFAULT;
        print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_results))

    if 'inference_method' not in locals():
        inference_method = INFERENCE_METHOD_DEFAULT;
        print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(inference_method))


    if 'block_size' not in locals():
        block_size = BLOCK_SIZE_DEFAULT;
        print('ATTENTION: The default value of %s for block_size is considered.\n' %str(block_size))
        
    if 'sparsity_flag' not in locals():
        sparsity_flag = SPARSITY_FLAG_DEFAULT;
        print('ATTENTION: The default value of %s for sparsity_flag is considered.\n' %str(sparsity_flag))
    
    if 'beta' not in locals():
        beta = BETA_DEFAULT
        print('ATTENTION: The default value of %s for beta is considered.\n' %str(beta))
    
    if 'alpha0' not in locals():
        alpha0 = ALPHA0_DEFAULT
        print('ATTENTION: The default value of %s for alpha0 is considered.\n' %str(alpha0))
        
    if 'bin_size' not in locals():
        bin_size = BIN_SIZE_DEFAULT
        print('ATTENTION: The default value of %s for bin_size is considered.\n' %str(bin_size))
    
    if 'no_neurons' not in locals():
        no_neurons = 0
        
    if 'class_sample_freq' not in locals():
        class_sample_freq = 0
        
    if 'kernel_choice' not in locals():
        print 'No kernel!'
        kernel_choice = 'E'
    if 'no_hidden_neurons' not in locals():
        no_hidden_neurons = 0
        
    #------------------------------------------------------------------------------

    #------------------Create the Necessary Directories if Necessary---------------
    if not os.path.isdir(file_name_base_results):
        os.makedirs(file_name_base_results)    
    if not os.path.isdir(file_name_base_results+'/Inferred_Graphs'):
        temp = file_name_base_results + '/Inferred_Graphs'
        os.makedirs(temp)
    if not os.path.isdir(file_name_base_results+'/Spent_Resources'):
        temp = file_name_base_results + '/Spent_Resources'
        os.makedirs(temp)
    if not os.path.isdir(file_name_base_results+'/Accuracies'):
        temp = file_name_base_results + '/Accuracies'
        os.makedirs(temp)            
    if not os.path.isdir(file_name_base_results+'/Plot_Results'):    
        temp = file_name_base_results + '/Plot_Results'
        os.makedirs(temp)    
    #------------------------------------------------------------------------------


    return no_stimul_rounds,no_neurons,file_name_data,file_name_base_results,inference_method,sparsity_flag,beta,alpha0,infer_itr_max,bin_size,no_processes,block_size,neuron_range,class_sample_freq,kernel_choice,no_hidden_neurons
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

def read_spikes_lines_integrated(file_name,line_no,n):
    
    a = linecache.getline(file_name, line_no)
    if a:
        a = (a[:-1]).split('\t')
        a = np.array(a)
        if len(a[0]):
            a = a.astype(float)
        else:
            a = np.zeros([n])#[]
            
        return a
            
    else:
        #pdb.set_trace()
        return np.zeros([n])#[]
    

def hinge_loss_func(x,FF,b,avg,lamb):
    temp = np.dot(FF,x) + b
    temp = np.multiply(temp,(temp>0).astype(int))
    #temp = avg*np.sum(temp) + lamb * pow(np.linalg.norm(x),2)
    temp = avg*np.sum(temp) + lamb * np.sum(np.abs(x))
    return temp



def l1_loss(x,a,b):
    return sum(np.abs(a.ravel()+x*b.ravel())) - x
#------------------------------------------------------------------------------


def calculate_integration_matrix(n_ind,spikes_file,n,t_start,t_end,tau_d,tau_s,kernel_choice,hidden_neurons):
    
    
    #----------------------------Initializations---------------------------
    t0 = math.log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 2/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    
    initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print 'initial memory is %s' %str(initial_memory)
    block_size = t_end - t_start
    
    if block_size < 0:
        return
    
    len_v = n + 1
        
    #X = np.zeros([len_v-1,block_size])
    V = np.zeros([len_v,block_size])
    Y = np.zeros([block_size])
    x = np.zeros([len_v,1])
    v = np.zeros([len_v,1])
    #----------------------------------------------------------------------
    
    #--------------------------Process the Spikes--------------------------
    t_tot = 0
    range_temp = range(t_start,t_end)
    
    for t in range_temp:

        #........Pre-compute Some of Matrices to Speed Up the Process......
        fire_t = read_spikes_lines(spikes_file,t,n)
        yy = -1
        if (n_ind in fire_t):                
            yy = 1
        #.................................................................
        
        #......................Read Incoming Spikes.......................
        fire_t = read_spikes_lines(spikes_file,t-1,n)
        fire_t = np.array(fire_t).astype(int)
        
        x = math.exp(-1/tau_s) * x
        x[fire_t] = x[fire_t] + 1
        
        v = math.exp(-1/tau_d) * v
        v[fire_t] = v[fire_t] + 1
        
        
        v[-1,0] = -1.0              # Thi corresponds to adding a column to larn the firing threshold
        #.................................................................
        
        #.....................Store the Potentials........................
        if kernel_choice == 'D':
            V[:,t_tot] = yy * v.ravel()#(np.delete(v,n_ind,0)).ravel()
        else:
            V[:,t_tot] = yy * (v-x).ravel()#(np.delete(v,n_ind,0)).ravel()
        #X[:,t_tot] = yy * (np.delete(x,n_ind,0)).ravel()
        Y[t_tot] = yy
        
        if yy>0:
            #~~~~~~~~~~~~~~~~Reset the Membrane Potential~~~~~~~~~~~~~~~~~
            x = 0*x#np.zeros([len_v,1])
            v = 0*v#np.zeros([len_v,1])
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        t_tot = t_tot + 1
        #.................................................................
        
    
    #-------------------Post Process Spike Matrices-----------------------
    Y = np.array(Y)
    V = V[:,0:t_tot]
    #X = X[:,0:t_tot]
    Y = Y[0:t_tot]
    
    #YY = (Y>0).astype(int) - (Y<=0).astype(int)
    #A = (V-X).T
    #AA = (V).T
    #---------------------------------------------------------------------
    
    #---------------Shift Post-Synaptic Spike One to the Left-------------
    #Y = np.roll(Y,-1)
    #Y[-1] = -1
    #Y[0] = -1
    #Y[1] = -1
    #---------------------------------------------------------------------
    

    #-------------------Delte Self History---------------------------------                    
    #AA = np.zeros(A.shape)
    #for t in range(0,t_tot):
    #    AA[t,:] = YY[t]*A[t,:]
                        
    #AA = np.delete(AA.T,n_ind,0).T
    
    V = V.T
    V = np.delete(V.T,n_ind,0).T
    
    if len(hidden_neurons):
        V = np.delete(V.T,hidden_neurons,0).T
    
    #---------------------------------------------------------------------
    
    flag_for_parallel_spikes = -1
    memory_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - initial_memory
    return V,Y,t_start,t_end,flag_for_parallel_spikes,memory_used


#------------------------------------------------------------------------------
#---------------------inference_constraints_hinge_parallel---------------------
# This function calculates the difference between rows of the firing activity
# matrix (A) and shifts the target firing activity one point to the left
# so that both are aligned.
#------------------------------------------------------------------------------
def calculate_difference_spike_matrix(A,Y):

    len_v,block_size = A.shape
    Y_New = Y

    A_New = np.zeros([len_v,block_size])

    #-------Calculate the Difference of the Firing Activity Matrix--------
    A_New[0:len_v-1,:-1] = np.diff(A) 
    A_New[len_v-1,:] = A[len_v-1,:]
    #---------------------------------------------------------------------

    #---------------Shift Post-Synaptic Spike One to the Left-------------
    if 0:
        Y = np.roll(Y,-1)
        Y[-1] = -1
        Y[0] = -1
        Y[1] = -1
    #---------------------------------------------------------------------

    return A_New,Y_New
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#---------------------inference_constraints_hinge_parallel---------------------
#------------------------------------------------------------------------------

def inference_constraints_hinge_parallel(out_spikes_tot_mat_file,TT,block_size,n,n_ind,num_process,inferece_params,hidden_neurons):

    max_memory = 0
    
    #----------------------Import Necessary Libraries----------------------
    from auxiliary_functions import soft_threshold
    from auxiliary_functions import soft_threshold_double
    import os.path
    
    import multiprocessing
    pool = multiprocessing.Pool(num_process)
    

    num_process_per_spike = int(max(multiprocessing.cpu_count(),num_process)/float(num_process))
    print multiprocessing.cpu_count()

    import gc
    #----------------------------------------------------------------------
    
    #---------------------Read Inference Parameters------------------------
    mthd = inferece_params[0]
    sparse_thr_0 = inferece_params[2]
    sparsity_flag = inferece_params[3]
    max_itr_opt = inferece_params[4]
    kernel_choice = inferece_params[10]
    
    #weight_of_weights = np.abs(np.array(range(0,n)) - n_ind)
    #weight_of_weights = np.exp(-weight_of_weights/(0.5*n))
    #inferece_params.append(weight_of_weights)
    #----------------------------------------------------------------------
    
    #----------------------------Initializations---------------------------    
    range_tau = range(0,max_itr_opt)
    
    T0 = 50                                    # It is the offset, i.e. the time from which on we will consider the firing activity
    
    print '-------------Neuron %s----------' %str(n_ind)
    #----------------------------------------------------------------------
    
    #-----------------------------Behavior Flags-------------------------------
    der_flag = 0                                # If 1, the derivative criteria will also be taken into account
    sketch_flag = 0                             # If 1, random sketching will be included in the algorithm as well
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV
    d_max = 10

    
    len_v = n+1-len(hidden_neurons)                 #The extra entry corresponds to larning the firing threshold 
    #--------------------------------------------------------------------------
    
    #---------------------Necessary Initializations------------------------    
    prng = RandomState(int(time.time()))
    block_start_inds = range(T0,TT,block_size)
    no_blocks = (1+TT-T0)/block_size
    
    if (mthd == 1) or (mthd == 2):
        lambda_tot = np.zeros([TT,1])
        dual_gap = np.zeros([len(range_tau),no_blocks])
        beta_K = 1
    else:
        beta_K = 1
    
    if (mthd == 3) or (mthd == 4):
        W_tot = np.random.randn(len_v-1,1)
        W_tot = W_tot - W_tot.mean()
        W_tot = whiten(W_tot)
        W_tot = W_tot/len_v
    else:
        W_tot = np.zeros([len_v-1,1])
        W_tot = np.random.randn(len_v-1,1)
        W_tot = W_tot - W_tot.mean()
        #W_tot = whiten(W_tot)
        W_tot = W_tot/(0.001+np.linalg.norm(W_tot))/float(len_v)
        
    Z_tot = np.zeros([len_v-1,1])    
        
    total_cost = np.zeros([len(range_tau)])
        
    A = np.zeros([block_size,len_v-1])      # This should contain current block
    YA = np.zeros([block_size])
    
    
    
    Delta_W = np.zeros([len_v-1,1])
    
    itr_block_t = 1
    itr_block_w = 0
    itr_cost = 0
    #--------------------------------------------------------------------------    
        
    #---Distribute Resources for Reading Spike Activity and Infer the Weights--
    num_process_sp = max(3,int(num_process/4.0))
    num_process_w = max(1,num_process - num_process_sp)
    t_step = int(block_size/float(num_process_sp))
    t_step_w = int(block_size/float(num_process_w))
    #--------------------------------------------------------------------------
    
    #--------------------Prepare the First Spike Matrix------------------------
    int_results = []
        
    for t_start in range(0,block_size,t_step):
        t_end = min(block_size-1,t_start + t_step)
        
        
        #calculate_integration_matrix(n_ind,out_spikes_tot_mat_file,n,t_start,t_end,tau_d,tau_s,kernel_choice)
        func_args = [n_ind,out_spikes_tot_mat_file,n,t_start,t_end,tau_d,tau_s,kernel_choice,hidden_neurons]
        int_results.append(pool.apply_async( calculate_integration_matrix, func_args) )
        #pool.close()
        #pool.join()
    
    max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    for result in int_results:
        (aa,yy,tt_start,tt_end,flag_spikes,memory_used) = result.get()
        max_memory = max_memory + memory_used
            
        A[tt_start:tt_end,:] = aa
        YA[tt_start:tt_end] = yy.ravel()
    #--------------------------------------------------------------------------

    # Use the differntial algorithm
    A[tt_start:tt_end,:],YA[tt_start:tt_end] = calculate_difference_spike_matrix(AA[tt_start:tt_end,:],YA[tt_start:tt_end])
    
    #---------------------------Infer the Connections--------------------------
    for ttau in range_tau:
            
        #~~~~~~~~~~~~~~~~~~~~~~~In-loop Initializations~~~~~~~~~~~~~~~~~~~~~~~~
        block_start_w = block_start_inds[itr_block_w]
        block_end_w = min(block_start_w + block_size,TT-1)
        int_results = []
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        #~~~~~~~~~~~~~~~~~Update theWeights Based on This Block~~~~~~~~~~~~~~~~
        if 0:
            func_args = [W_tot,A,YA,lambda_tot[block_start_w:block_end_w],len_v,block_start_w,block_end_w,inferece_params]
            int_results.append(pool.apply_async(infer_w_block, func_args) )
            t_end_last_w = block_end_w
        else:
            for t_start in range(block_start_w,block_end_w,t_step_w):
                t_end_w = min(t_start + t_step_w,block_end_w-1)
                    
                if t_end_w - t_start < 10:
                    continue
                
                if (mthd == 1) or (mthd == 2):
                    lambda_temp = lambda_tot[t_start:t_end_w]
                else:
                    lambda_temp = []
        
                inferece_params[10] = 0
                #Delta_W,d_alp_vec,t_start,t_end,cst,memory_used  = infer_w_block(W_tot,A,YA,lambda_temp,len_v,0,t_step_w,inferece_params)
                
                #pdb.set_trace()
                    
                if 0:#not (itr_cost%2):
                    func_args = [np.zeros([len_v-1,1]),A[t_start-block_start_w:t_end_w-block_start_w,:],YA[t_start-block_start_w:t_end_w-block_start_w],lambda_temp,len_v,t_start,t_end_w,inferece_params]
                else:
                    func_args = [W_tot,A[t_start-block_start_w:t_end_w-block_start_w,:],YA[t_start-block_start_w:t_end_w-block_start_w],lambda_temp,len_v,t_start,t_end_w,inferece_params]
                int_results.append(pool.apply_async(infer_w_block, func_args) )
                t_end_last_w = t_end_w
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #~~~~~~~~~~~~~~~~~Process the Spikes for the Next Block~~~~~~~~~~~~~~~~
        block_start = block_start_inds[itr_block_t]
        block_end = min(block_start + block_size,TT-1)
            
        for t_start in range(block_start,block_end,t_step):
            t_end = min(t_start + t_step,block_end-1)
                
            if t_end - t_start < 10:
                continue
                        
            func_args = [n_ind,out_spikes_tot_mat_file,n,t_start,t_end,tau_d,tau_s,kernel_choice,hidden_neurons]
            int_results.append(pool.apply_async( calculate_integration_matrix, func_args) )
            t_end_last_t = t_end
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        total_memory_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print 'memory so far up to iterations %s is %s' %(str(ttau),str(total_memory_init))
            
        #~~~~~~~~~~~~~~~~~~~~Retrieve the Processed Results~~~~~~~~~~~~~~~~~~~~
        mem_temp = 0
        for result in int_results:
            (aa,yy,tt_start,tt_end,spike_flag,memory_used) = result.get()
            mem_temp = mem_temp + memory_used
            if spike_flag < 0:
                A[tt_start-block_start:tt_end-block_start,:] = aa
                YA[tt_start-block_start:tt_end-block_start] = yy
                
                # Use the differntial algorithm
                A[tt_start-block_start:tt_end-block_start,:],YA[tt_start-block_start:tt_end-block_start] = calculate_difference_spike_matrix(aa,yy) 

                if tt_end == t_end_last_t:
                    itr_block_t = itr_block_t + 1
                    if itr_block_t >= len(block_start_inds):
                        itr_block_t = 0
                            
            else:
                Delta_W_loc = aa            # This is because of the choice of symbols for result.get()
                d_alp_vec = yy              # This is because of the choice of symbols for result.get()
                cst = spike_flag            # This is because of the choice of symbols for result.get()
                    
                Delta_W = Delta_W + Delta_W_loc
                    
                if tt_end == t_end_last_w:
                    itr_block_w = itr_block_w + 1
                    total_cost[itr_cost] = total_cost[itr_cost] + sum(np.dot(A,W_tot)<0)
                    print total_cost[itr_cost]
                
                if (mthd == 1) or (mthd == 2):
                    lambda_tot[tt_start:tt_end] = lambda_tot[tt_start:tt_end] + d_alp_vec * (beta_K/float(no_blocks)) 
        
        #pdb.set_trace()       
        if itr_block_w >= len(block_start_inds):
            #Delta_W_loc = np.dot(A.T,lambda_tot[b_st:t_end_last_w+2])
            
            W_tot = W_tot + (beta_K/float(no_blocks)) * np.reshape(Delta_W,[len_v-1,1])
            
            if (mthd == 3) or (mthd == 4) or (mthd == 6):
                W_tot[:-1] = W_tot[:-1] - W_tot[:-1].mean()
                
            W_tot = W_tot/(0.0001+np.linalg.norm(W_tot))
            if sparsity_flag:
                sparse_thr = W_tot[:-1].std()/float(sparse_thr_0)
                if sparse_thr!=0:
                    sparse_thr_pos = np.multiply(W_tot[:-1],(W_tot[:-1]>=0).astype(int)).std()/float(sparse_thr_0)
                    sparse_thr_neg = np.multiply(W_tot[:-1],(W_tot[:-1]<0).astype(int)).std()/float(sparse_thr_0)
                    W_tot[:-1] = soft_threshold_double(W_tot[:-1],sparse_thr_pos,sparse_thr_neg)
                #W_tot[:-1] = soft_threshold(W_tot[:-1],sparse_thr)
                
                
            print 'Processing %s blocks was finished, with cost being %s' %(str(no_blocks),str(total_cost[itr_cost]))
            #W_tot = np.multiply(W_tot,(W_tot>0).astype(int))
            itr_block_w = 0
            itr_cost = itr_cost + 1
            Delta_W = 0*Delta_W#np.zeros([n,1])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        total_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + mem_temp
        if total_memory>max_memory:
            max_memory = total_memory
            
        print 'Total memory after iteration %s is %s' %(str(ttau),str(max_memory))
            
        #~~~~~~~~~~Break If Stopping Condition is Reached~~~~~~~~~~~
        if itr_cost >= 7:
            if abs(total_cost[itr_cost-1]-total_cost[itr_cost-2])/(0.001+total_cost[itr_cost-2]) < 0.00001:
                break
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        
    print total_cost[1:itr_cost]
    #--------------------------------------------------------------------
        
    
    WW = np.zeros([len_v,1])
    WW[0:n_ind,0] = W_tot[0:n_ind,0]
    WW[n_ind+1:,0] = W_tot[n_ind:,0]
    
    A = None
    YA = None
    
    pool.close()
    pool.join()
    
    
    
    return WW[0:len_v].ravel(),max_memory,total_cost[1:itr_cost]
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def infer_w_block(W_in,aa,yy,lambda_temp,len_v,t_start,t_end,inferece_params):
    
    warnings.filterwarnings("error")
    #---------------------Read Inference Parameters------------------------
    mthd = inferece_params[0]
    alpha0 = inferece_params[1]
    sparse_thr_0 = inferece_params[2]
    sparsity_flag = inferece_params[3]    
    max_itr_opt = inferece_params[4]
    beta = inferece_params[6]
    class_sample_freq = inferece_params[8]
    rand_sample_flag = inferece_params[9]
    #weights_weight = inferece_params[12]
    
    #weights_weight = np.reshape(weights_weight,[len_v-1,1])
    #----------------------------------------------------------------------
    
    #------------------------Initializations------------------------
    initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    from auxiliary_functions import soft_threshold
    from auxiliary_functions import soft_threshold_double
    from numpy.random import RandomState
    prng = RandomState(int(time.time()))
    #if not np.linalg.norm(W_in):
    #    W_in = np.random.rand(W_in.shape[0],W_in.shape[1])
    #    W_in = W_in/np.linalg.norm(W_in)
    
    if rand_sample_flag:
        t_gap = 4
        #t_init = np.random.randint(0,t_gap)
        t_init = prng.randint(0,t_gap)
        t_inds = np.array(range(t_init,t_end-t_start,t_gap))
        aa = aa[t_inds,:]
        yy = yy[t_inds]
        
    #---------------------------------------------------------------
    
    #--------------Initialize Simulation Parameters-----------------
    TcT = len(yy)
        
    if TcT == 0:
        print 'error! empty activity file.'
        return
    
    lamb = .1/float(TcT)
    max_internal_itr = 15*TcT
    
    cf = lamb*TcT
    ccf = 1/float(cf)
    
    cst = 0
    cst_y = 0
    cst_old = 0
    
    
    ind_ones = np.nonzero(yy>0)[0]
    ind_zeros = np.nonzero(yy<0)[0]
    no_ones = sum(yy>0)
    no_zeros = sum(yy<0)
    p1 = no_ones /float(len(yy))
    
    #pdb.set_trace()
    
    W_temp = copy.deepcopy(W_in)
    Delta_W = np.zeros(W_temp.shape)
    Momentum_W = np.zeros(W_temp.shape)
    mu_moment = 0.9
    eta_w = 0.001
    #W_temp[-1] = 0.1
    
    no_firings_per_neurons = 2*np.ones(W_temp.shape)                # This variable tracks the number of times each pre-synaptic neuron has fired
    if 0:#max(W_temp.shape) == 94:
        no_firings_per_neurons = [ 0.51390311,  0.55878293,  0.56535145,  0.5646233 ,
             0.56620095,  0.56255262,  0.62126728,  0.53129527,  0.58542108,
             0.58517077,  0.50000758,  0.5909808 ,  1.12309145,  0.5       ,
             0.50772901,  0.5       ,  0.51575382,  0.50471022,  0.54981758,
             0.51781692,  0.55744799,  0.51950076,  0.50467988,  0.51113462,
             0.50034132,  0.51501809,  0.50334494,  0.52704773,  0.52007721,
             0.50667471,  0.51799137,  0.50101638,  0.50304154,  0.50472539,
             0.50543837,  0.66023088,  0.84892029,  0.52372555,  0.60291943,
             0.53981311,  0.50069023,  0.50150181,  0.51010308,  0.50361799,
             0.50138804,  0.5000986 ,  0.51975865,  0.5071298 ,  0.52157144,
             0.50666712,  0.50118324,  0.51463885,  0.51724805,  0.50744837,
             0.51783967,  0.57304253,  0.87852413,  0.51447956,  0.52308842,
             0.55449746,  0.57268604,  0.50339045,  0.50276849,  0.50352698,
             0.50037166,  0.50409584,  0.52183691,  0.6436048 ,  1.5       ,
             1.40827588,  0.51577658,  0.50796414,  0.51709635,  0.94915466,
             0.60803164,  0.54299118,  0.52094948,  0.51003481,  0.65719693,
             0.60683323,  0.51222685,  0.5372039 ,  0.51454024,  0.51114221,
             0.73898484,  0.50840406,  0.50345871,  0.50039441,  0.50129702,
             0.50211619,  0.50580244,  0.502814  ,  0.50370901, 1]
        
    no_firings_per_neurons = np.array(no_firings_per_neurons)
    no_firings_per_neurons = np.reshape(no_firings_per_neurons,[len(no_firings_per_neurons),1])
    #no_firings_per_neurons = np.delete(W_r,hidden_neurons,0)
    #---------------------------------------------------------------
    
    #-----------------Adjust Hinge Loss Parameters------------------
    e0 = 0.2                    # The offset: L(x) = h0*max(0,e0-x)
    h0 = 1                      # The slope: L(x) = h0*max(0,e0-x)
    
    if mthd == 5:
        e1 = 5                  # The other offset for double hinge loss: L(x) = h0*max(0,e0-x) + h1*max(0,x-e1)
        h1 = 1                  # The other slope for double hinge loss: L(x) = h0*max(0,e0-x) + h1*max(0,x-e1)
    #---------------------------------------------------------------
    
    #----------------------Assign Dual Vectors----------------------
    if (mthd == 1) or (mthd == 2):
        d_alp_vec = np.zeros([len(lambda_temp),1])
        if rand_sample_flag:
            lambda_temp = lambda_temp[t_inds]
    else:
        d_alp_vec = [0]
    #---------------------------------------------------------------
        
    #------------------------Infer the Weights----------------------
    if mthd == 10:
        yy = np.reshape(yy,[1,TcT])
        Delta_W = np.dot(yy,aa)
        Delta_W = Delta_W.T/(0.00001 + np.linalg.norm(Delta_W))
    else:    
        if no_ones and no_zeros: 
            
            #--------------------Do One Pass over Data----------------------        
            for ss in range(0,max_internal_itr):
                
                #~~~~~~Sample Probabalistically From Unbalanced Classes~~~~~
                try:
                    if class_sample_freq:
                        ee = np.random.rand(1)
                        if ee < class_sample_freq:
                            #ii = np.random.randint(0,no_ones)
                            ii = prng.randint(0,no_ones)
                            jj = ind_ones[ii]
                        else:
                            #ii = np.random.randint(0,no_zeros)
                            ii = prng.randint(0,no_zeros)
                            jj = ind_zeros[ii]
                    else:
                        #ii = np.random.randint(0,TcT)
                        ii = prng.randint(0,TcT)
                        jj = ii
                except:
                    print 'something is fishy: ee = %s,no_ones = %s,no_zeros=%s, ii = %s,jj=%s' %(str(ee),str(no_ones),str(no_zeros),str(ii),str(jj))
                    continue
                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
                
                #~~~~~~~~~~~~~~~~~~~~~~Retrieve a Vector~~~~~~~~~~~~~~~~~~~~
                
                aa_t = aa[jj,:]#/float(cf)
                yy_t = yy[jj]#[0]
                #no_firings_per_neurons = no_firings_per_neurons + np.reshape(((yy_t*aa_t.ravel())>0.9).astype(int),[len_v-1,1])
                aa_t = aa_t/(0.00001+np.linalg.norm(aa_t))
                if yy_t * sum(aa_t[:-1])<0:
                    print 'something bad is happening!'
                    #pdb.set_trace()
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~~~~~~Update Variables for SDCS~~~~~~~~~~~~~~~~
                if (mthd == 1) or (mthd == 2):
                    #c = 1 * yy_t
                    #gamma_t = 0.5* ( (1+yy_t) * no_ones + (1-yy_t) * no_zeros)
                    #ccf = gamma_t
                    ub = h0-lambda_temp[jj]
                    lb = -lambda_temp[jj]
                    
                    if 0:
                        if yy_t>0:
                            ub = h0-lambda_temp[jj]
                            lb = -lambda_temp[jj]
                        else:
                            lb = -h0-lambda_temp[jj]
                            ub = -lambda_temp[jj]
                            
                    
                    #b = cf * (e0-np.dot(W_temp.T,aa_t))/pow(np.linalg.norm(aa_t),2)
                    b = (e0-np.dot(W_temp.T,aa_t))#/pow(np.linalg.norm(aa_t),2)
                    #b = yy_t * b
                    d_alp = min(ub,max(lb,b))
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~Update Variables for SDCS with Double Hinge Loss~~~~~
                elif (mthd == 5):
                    lb0 = -lambda_temp[jj]
                    ub0 = h0-lambda_temp[jj]
                    lb1 = -lambda_temp[jj]-h1
                    ub1 = -lambda_temp[jj]
                    
                    b0 = (e0-np.dot(W_temp.T,aa_t))#/pow(np.linalg.norm(aa_t),2)
                    b1 = (e1-np.dot(W_temp.T,aa_t))#/pow(np.linalg.norm(aa_t),2)
                    
                    d_alp0 = min(ub0,max(lb0,b0))
                    val0 = pow(d_alp0-b0,2)
                    d_alp1 = min(ub1,max(lb1,b1))
                    val1 = pow(d_alp1-b1,2)
                    
                    if val1 < val0:
                        d_alp = d_alp1
                    else:
                        d_alp = d_alp0
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~~~~~~Upate Weights for SDCD~~~~~~~~~~~~~~~~~~~
                if (mthd == 1):
                    Delta_W_loc = d_alp * np.reshape(aa_t,[len_v-1,1])# * yy_t#/float(cf)
                    #Delta_W_loc = np.divide(Delta_W_loc,0.5*(no_firings_per_neurons))
                    #Delta_W_loc = np.divide(Delta_W_loc,(no_firings_per_neurons))
                    
                    #Delta_W_loc = np.multiply(Delta_W_loc,(weights_weight))
                    #Delta_W_loc = np.divide(Delta_W_loc,(no_firings_per_neurons))
                    if d_alp !=0:
                        s_size = max(0,e0-np.dot(W_temp.T,aa_t)) * yy_t /(d_alp)
                        s_size = max(0,1.0001*e0-np.dot(W_temp.T,aa_t)) /(d_alp)
                    else:
                        s_size = 1
                    
                    if 0:                
                        sparse_thr_pos = np.multiply(W_temp[:-1],(W_temp[:-1]>=0).astype(int)).std()/float(sparse_thr_0)
                        sparse_thr_neg = np.multiply(W_temp[:-1],(W_temp[:-1]<0).astype(int)).std()/float(sparse_thr_0)
                        sparse_thr = W_temp[:-1].std()/float(sparse_thr_0)
                        
                        W_temp[-1] = W_temp[-1] + s_size * Delta_W_loc[-1]
                        #W_temp[:-1] = soft_threshold_double(W_temp[:-1],sparse_thr_pos,sparse_thr_neg) + s_size * Delta_W_loc[:-1]
                        W_temp[:-1] = soft_threshold(W_temp[:-1],sparse_thr) + s_size * Delta_W_loc[:-1]
    
                    else:
                        W_temp = W_temp + s_size * Delta_W_loc
                        
                    
                    Delta_W = Delta_W + s_size * Delta_W_loc
                    #cst = np.dot(aa,W_temp);cst[jj-20:jj+20].T
                    #cst = np.dot(aa,Delta_W_loc);cst[jj-20:jj+20].T
                
                    #0.05*yy[jj-20:jj+20].T
                    #pdb.set_trace()
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~Upate Weights for Sketched SDCD~~~~~~~~~~~~~~~
                elif (mthd == 2):
                    Delta_W_loc = d_alp * np.reshape(aa_t,[len_v-1,1])# * yy_t#/float(cf)
                    Delta_W_loc = np.divide(Delta_W_loc,0.1*np.log(no_firings_per_neurons))
                    
                    #s = prng.randint(0,beta,[len_v-1,1])
                    #s = (s>=beta-1).astype(int)
                    #Delta_W_loc = np.multiply(Delta_W_loc,s)
                    #Delta_W_loc = Delta_W_loc *pow(np.linalg.norm(aa_t),2) /(0.0001+pow(np.linalg.norm(np.multiply(np.reshape(aa_t,[len_v-1,1]),s)),2))
                    
                    
                    if d_alp !=0:
                        s_size = max(0,e0-np.dot(W_temp.T,aa_t)) * yy_t /(d_alp)
                        s_size = max(0,e0-np.dot(W_temp.T,aa_t)) /(d_alp)
                    else:
                        s_size = 1
                    
                    s_size = 1
                    Delta_W = Delta_W + s_size * Delta_W_loc
                    W_temp = W_temp + s_size * Delta_W_loc
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            
                #~~~~~~~~~~~~~~Upate Weights for Perceptron~~~~~~~~~~~~~~~~~
                elif (mthd == 3):
                    #Delta_W_loc = np.reshape(aa_t,[len_v-1,1]) * 0.5 * (np.sign(xx-1) + np.sign(xx-10)))
                    
                    
                    d_alp = max(0,0.5*e0-np.dot(W_temp.T,aa_t))
                    Delta_W_loc = 0
                    if d_alp:
                        Delta_W_loc = max(0,e0-np.dot(W_temp.T,aa_t))*np.reshape(aa_t,[len_v-1,1])
                    
                    #Delta_W_loc = Delta_W_loc + 0.7*(W_temp<0).astype(int)
                    Delta_W_loc = 1*Delta_W_loc - 0.001*W_temp
                    
                    #Delta_W_loc = -Delta_W_loc 
                    #Momentum_W = mu_moment * Momentum_W - eta_w * Delta_W_loc
                        
                    
                    #Delta_W_loc[-1] = .1
                    Delta_W = Delta_W + Delta_W_loc
                    #W_temp = W_temp + 1*Momentum_W
                    W_temp = W_temp + 1*Delta_W_loc
                    #W_temp = W_temp - W_temp.mean()
                    #pdb.set_trace()
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~Upate Weights for Skecthed Perceptron~~~~~~~~~~~
                elif (mthd == 4):
                    d_alp = max(0,0.5*e0-np.dot(W_temp.T,aa_t))
                    d_alp = max(0,0-np.dot(W_temp.T,aa_t))
                    if d_alp:
                        try:
                            s = prng.randint(0,beta,[len_v-1,1])
                            s = (s>=beta-1).astype(int)
                            
                            
                            Delta_W_loc = max(0,e0-np.dot(W_temp.T,aa_t))*np.reshape(aa_t,[len_v-1,1])
                            #Delta_W_loc = np.multiply(Delta_W_loc,s)
                            
                            #Delta_W_loc = np.divide(Delta_W_loc,0.4*np.log(no_firings_per_neurons))
                            #pdb.set_trace()
                            Delta_W_loc = np.divide(Delta_W_loc,(no_firings_per_neurons))
                            #Delta_W_loc = np.multiply(Delta_W_loc,(weights_weight))
                    
                            if 0:#sum(s):
                                Delta_W_loc = Delta_W_loc/(0.0001+pow(np.linalg.norm(np.multiply(np.reshape(aa_t,[len_v-1,1]),s)),2))
                            
                            if 1:
                                sparse_thr_pos = np.multiply(W_temp[:-1],(W_temp[:-1]>=0).astype(int)).std()/float(sparse_thr_0)
                                sparse_thr_neg = np.multiply(W_temp[:-1],(W_temp[:-1]<0).astype(int)).std()/float(sparse_thr_0)
                                #sparse_thr = W_temp[:-1].std()/float(sparse_thr_0)
                                
                                W_temp[-1] = W_temp[-1] + 0.1 * Delta_W_loc[-1]
                                #W_temp[:-1] = soft_threshold_double(W_temp[:-1],sparse_thr_pos,sparse_thr_neg) + 0.1 * Delta_W_loc[:-1]
                        except RuntimeWarning:
                            pdb.set_trace()
                        
                        #Delta_W_loc = 1*Delta_W_loc - 0.001*W_temp
                        #Delta_W_loc[-1] = .1
                        Delta_W = Delta_W + Delta_W_loc
                        Delta_W = Delta_W/(0.00001+ np.linalg.norm(Delta_W))
                        #W_temp = W_temp + 1*Delta_W_loc
                        #W_temp = W_temp - W_temp.mean()
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~Upate Weights for Perceptron with Double Hinge Loss~~~~
                elif (mthd == 6):
                    #Delta_W_loc = np.reshape(aa_t,[len_v-1,1]) * 0.5 * (np.sign(xx-1) + np.sign(xx-10)))
                    d_alp = max(0,0.5*e0-np.dot(W_temp.T,aa_t))
                    if d_alp:
                        if np.dot(W_temp.T,aa_t)<e0:
                            Delta_W_loc = max(0,e0-np.dot(W_temp.T,aa_t))*np.reshape(aa_t,[len_v-1,1])#/(0.0001+pow(np.linalg.norm(aa_t),2))
                        else:
                            Delta_W_loc = -max(0,e1+np.dot(W_temp.T,aa_t))*np.reshape(aa_t,[len_v-1,1])#/(0.0001+pow(np.linalg.norm(aa_t),2))
                        
                        Delta_W_loc = 1*Delta_W_loc - 0.001*W_temp
                        #Delta_W_loc[-1] = .1
                        Delta_W = Delta_W + Delta_W_loc
                        W_temp = W_temp + 1*Delta_W_loc
                        #W_temp = W_temp - W_temp.mean()
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                
                #~~~Upate Weights for Perceptron with Cross Entropy Loss~~~~
                elif (mthd == 7):
                    x_t = np.reshape(aa_t,[len_v-1,1]) * yy_t
                    yyy_t = (yy_t + 1)*0.5
                    z_t = 1/(1.0 + math.exp(-(np.dot(W_temp.T,x_t))))
                    Delta_W_loc = x_t * (z_t - yyy_t)
                    
                    Delta_W_loc = -1*Delta_W_loc - 0.001*W_temp
                    Delta_W_loc = 0.005*Delta_W_loc
                    #Delta_W_loc[-1] = .1
                    Delta_W = Delta_W + Delta_W_loc
                    W_temp = W_temp + 1*Delta_W_loc
                    #W_temp = W_temp - W_temp.mean()
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~Update Dual Vectors If Necessarry~~~~~~~~~~~~~
                if (mthd == 1) or (mthd == 2) or (mthd == 5):
                    lambda_temp[jj] = lambda_temp[jj] + d_alp
                    if rand_sample_flag:
                        d_alp_vec[t_inds[jj]] = d_alp_vec[t_inds[jj]] + d_alp
                    else:
                        d_alp_vec[jj] = d_alp_vec[jj] + d_alp
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #~~~~~~~~~~~~~~~~~~~~~~~Update Costs~~~~~~~~~~~~~~~~~~~~~~~~
                #W_temp = np.multiply(W_temp,(W_temp>0).astype(int))
                if 0:#sparsity_flag:                
                    sparse_thr_pos = np.multiply(W_temp[:-1],(W_temp[:-1]>=0).astype(int)).std()/float(sparse_thr_0)
                    sparse_thr_neg = np.multiply(W_temp[:-1],(W_temp[:-1]<0).astype(int)).std()/float(sparse_thr_0)
                    W_temp[:-1] = soft_threshold_double(W_temp[:-1],sparse_thr_pos,sparse_thr_neg)
                    
                cst = cst + np.sign(max(0,.1-np.dot(W_temp.T,aa_t)))#(hinge_loss_func(W_temp,-aa_t,.1,1,0))
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #--------------In Compliance with Jaggi's Work------------------
            #Delta_W = np.dot(aa.T,d_alp_vec)
            #---------------------------------------------------------------
            
        #-------------------------------------------------------------------
        else:
            print 'no ones!'
        
        w_flag_for_parallel = -1                # This is to make return arguments to 4 and make sure that it is distinguishable from other parallel jobs
        #pdb.set_trace()
    memory_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - initial_memory
    if no_firings_per_neurons.max() > 2:
        print no_firings_per_neurons
    return Delta_W,d_alp_vec,t_start,t_end,cst,memory_used

#------------------------------------------------------------------------------
