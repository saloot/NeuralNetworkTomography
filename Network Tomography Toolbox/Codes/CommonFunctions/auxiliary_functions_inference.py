#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
#from brian import *
from scipy import sparse,linalg
import warnings
import pdb,os,sys
import random
import copy
import numpy.ma as ma
import matplotlib.pyplot as plt
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
                no_stimul_rounds = int(arg)                         # Number of times we inject stimulus to the network
            elif opt == '-A':
                file_name_data = str(arg)                           # The folder to store results
            elif opt == '-Q':
                no_processes = int(arg)                             # The number of cpu cores to use for simulaitons
            elif opt == '-N':
                no_neurons = int(arg)                               # Number of observed eurons
            elif opt == '-M':
                inference_method = int(arg)                         # The inference method
            elif opt == '-B':
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


#==============================================================================
#===========================read_integration_matrix============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function gets the file a an imput and a set of lines, then reads the
# integrated activity detais stored between the specified lines.

# INPUTS:
#   file_name: name (path) of the file where the data is stored
#   start_line: the starting line that we wish to read from
#   end_line: the ending line that we wish to read from
#   n: number of neuron (size of the network)

# OUTPUTS:
#   V: the integration data matrix with size (end_line - start_line) * n
#------------------------------------------------------------------------------

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



def hinge_loss_func(x,FF,b,avg,lamb):
    temp = np.dot(FF,x) + b
    temp = np.multiply(temp,(temp>0).astype(int))
    #temp = avg*np.sum(temp) + lamb * pow(np.linalg.norm(x),2)
    temp = avg*np.sum(temp) + lamb * np.sum(np.abs(x))
    return temp



def l1_loss(x,a,b):
    
    return sum(np.abs(a.ravel()+x*b.ravel())) - x


#------------------------------------------------------------------------------

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def detect_spike_peaks(V,n,t_fire):
    from scipy.cluster.vq import vq
    from scipy import signal
    # V is the membrane potential
    # n is the number of peaks to detect
    # t_fire contains the actual spiking times.
    
    
    n_peak = 0          # The number of counted peaks so far
    peak_values = []    # The magnitude of the detected peaks
    peak_inds = []      # Indices of the peaks
    U = copy.deepcopy(V)    
    U = np.multiply(U,(U>0).astype(int))
    
    U = smooth(U.ravel(),window_len=15,window='hanning')
    #peakind = signal.find_peaks_cwt(U, np.arange(1,20))
    #peak_vals = U[peakind]
    
    #U = np.reshape(U,[len(U),1])
    
    if 1:
        while n_peak < n:
    
            peakind = signal.find_peaks_cwt(U, np.arange(1,5))
            peak_vals = U[peakind]
            ind_max = np.argmax(peak_vals)
            ind_max = peakind[ind_max]
            p_max = np.max(peak_vals)   
            peak_inds.append(ind_max)
            peak_values.append(p_max)
            
            #~~~~~~~~~Reset Membrane Potential~~~~~~~
            aa = np.diff(U[ind_max+1:])
            aa = np.nonzero(aa>0)[0]
            aa = aa[0]
            delta_val = U[aa-1]
            U[ind_max+1:aa] = 0
            U[aa:] = U[aa:] - delta_val
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #~~~~~~~~~~Remove Current Peak~~~~~~~~~~~
            aa = np.diff(U[ind_max-100:ind_max])
            for ii in range(len(aa)-1,-1,-1):
                if aa[ii] < 0:
                    break
            U[ind_max-ii:ind_max] = 0
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if 0:
                temp_fire_orig = copy.deepcopy(t_fire).tolist()
                temp_fire_orig.append(ind_max)
                temp_fire_orig = np.array(temp_fire_orig)
                temp_fire_orig.sort()
                temp_fire_orig = temp_fire_orig.tolist()
                ind1 = temp_fire_orig.index(ind_max)
                t_fire_next = t_fire[ind1]
                t_fire_bef = min(ind_max,t_fire[max(ind1-1,0)])
                
                U[t_fire_bef+1:t_fire_next+1] = U[t_fire_bef+1:t_fire_next+1] - p_max
            
            U[ind_max] = 0
            
            n_peak = n_peak + 1
            print n_peak
        
    pdb.set_trace()
    
    if 0:
        code_book= range(0,int(1000*U.max()),int(1000*theta))
        code_book = np.reshape(code_book,[len(code_book),1])
        code_book = np.array(code_book)/1000.0
        peakind = signal.find_peaks_cwt(data, np.arange(1,20))
        U = np.multiply(U,(U>0).astype(int))
        pdb.set_trace()
        U = smooth(U.ravel(),window_len=11,window='hanning')
        U = np.reshape(U,[len(U),1])
        theta = (U.max() - 0)/float(n)
        U_quant = vq(U,code_book)
        U_quant = np.diff(U_quant[0])
        U_quant = np.multiply(U_quant,(U_quant>0).astype(int))
        
    
    return peak_inds,peak_values
        
        
        
        
    
    pass


#------------------------------------------------------------------------------
def spike_pred_accuracy(out_spikes_tot_mat_file,T_array,W,n_ind,theta):
    
    #----------------------------Initilizations--------------------------------    
    n = max(np.shape(W)) -1 
    #--------------------------------------------------------------------------
    
    #-----------------------------Behavior Flags-------------------------------
    der_flag = 0                                # If 1, the derivative criteria will also be taken into account
    rand_sample_flag = 1                        # If 1, the samples will be wide apart to reduce correlation
    sketch_flag = 0                             # If 1, random sketching will be included in the algorithm as well
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV)
    delta = 0.25                                       # The tanh coefficient to approximate the sign function
    d_max = 10
    t_gap = 3                                    # The gap between samples to consider
    t_avg = 1
    block_size = 20000
    bin_size = 1
    
    t0 = math.log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 2/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    #--------------------------------------------------------------------------
    
    #---------Identify Incoming Connections to Each Neuron in the List---------
    opt_score_true_pos = 0
    opt_score_true_neg = 0
    for T_pair in T_array:
        
        range_T = range(max(T_pair[0]-1000,0),T_pair[1])
        T_temp = len(range_T)
        T0 = T_pair[0]
        
        print '-------------T is from %s to %s----------' %(str(T_pair[0]),str(T_pair[1]))
    
    
        
        #~~~~~~~~~~~~~~~~~Calculate The Initial Inverse Matrix~~~~~~~~~~~~~~~~~
        X = np.zeros([n+1,int(T_temp/float(t_avg))])
        V = np.zeros([n+1,int(T_temp/float(t_avg))])
        x = np.zeros([n+1,1])
        v = np.zeros([n+1,1])
        xx = np.zeros([n+1,1])
        vv = np.zeros([n+1,1])
        Y = np.zeros([int(T_temp/float(t_avg))])
        
        yy = 0
        
        t_counter = 0
        t_tot = 0
        for t in range_T:
            
            #........Pre-compute Some of Matrices to Speed Up the Process......
            #fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
            if (n_ind in fire_t):                
                yy = 1
                #x = np.zeros([n+1,1])
                #v = np.zeros([n+1,1])
            else:
                yy = 0
            
            fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
            x = math.exp(-1/tau_s) * x
            
            fire_t = np.array(fire_t).astype(int)
            x[fire_t] = x[fire_t] + 1
            
            v = math.exp(-1/tau_d) * v
            v[fire_t] = v[fire_t] + 1
            
            #v[-1,0] = -1
                
            V[:,t_tot] = v.ravel()
            X[:,t_tot] = x.ravel()
            Y[t_tot] = yy
                
            t_tot = t_tot + 1
            
        Y = np.array(Y)
        
        V = V[:,0:t_tot]
        X = X[:,0:t_tot]
        Y = Y[0:t_tot]
        
        
        #A = (V-X).T
        A = (V).T
        #A = (V-X).T
        #A = (A>0.85).astype(int)
        A = A[T_pair[0] - range_T[0]:T_pair[1],:]
        Y = Y[T_pair[0] - range_T[0]:T_pair[1]]
        A_orig = copy.deepcopy(A)
        Y_orig = copy.deepcopy(Y)
        
        
        
        #--------Shift Post-Synaptic Spike One to the Left---------
        Y_orig = np.roll(Y_orig,-1)
        Y_orig[-1] = -1
        #----------------------------------------------------------
        
        if rand_sample_flag:
            t_init = np.random.randint(0,t_gap)
            t_inds = np.array(range(t_init,T_pair[1]-T_pair[0],t_gap))

            A = A[t_inds,:]
            Y = Y_orig[t_inds]
            
        
        #--------------Calculate Prediction Accuracy----------------        
        Y_predict = np.dot(A_orig,W)
        #plt.plot(Y_predict[0:1000]);plt.plot(Y_orig[0:1000],'g');plt.show()
        # aa = -np.ones(W.shape)
        # aa = aa/np.linalg.norm(aa)
        #Y_predict2 = np.dot(A_orig,aa)
        #plt.plot(Y_orig[1000:2000]);plt.plot(Y_predict[1000:2000],'r');plt.show()
        #plt.plot(Y_orig[1000:2000]);plt.plot(Y_predict2[1000:2000],'r');plt.show()
        #plt.plot(Y_orig[1000:2000]);plt.plot(Y_predict[1000:2000],'r');plt.plot(Y_predict2[1000:2000],'g');plt.show()
        #pdb.set_trace()
        
        Y_orig = np.reshape(Y_orig,[len(Y_orig),1])
        
        
        if bin_size > 1:
            Y_predict = (Y_predict>=theta).astype(int)
            Y_predict = np.reshape(Y_predict,[len(Y_predict),1])
            ll = int(len(Y_predict)/float(bin_size))
            
            Y_orig_binned = np.reshape(Y_orig,[ll,bin_size])
            Y_orig_binned = Y_orig_binned.mean(axis = 1)
            Y_orig_binned = (Y_orig_binned>0).astype(int)
            
            Y_predict_binned = np.reshape(Y_predict,[ll,bin_size])
            
            Y_predict_binned = Y_predict_binned.mean(axis = 1)
            
            Y_predict_binned = (Y_predict_binned>0).astype(int)
            
            
        
            temp = np.multiply((Y_predict_binned==1).astype(int),(Y_orig_binned==1).astype(int))
            opt_score_true_pos = opt_score_true_pos + sum(temp)/(sum(Y_orig_binned==1)+0.0001)
            
            temp = np.multiply((Y_predict_binned==0).astype(int),(Y_orig_binned==0).astype(int))
            opt_score_true_neg = opt_score_true_neg + sum(temp)/(sum(Y_orig_binned==0)+0.0001)
            
            #plt.plot(Y_orig_binned);plt.plot(Y_predict_binned,'r');plt.show()
            
        else:
            
            ll = int(100 * Y_predict.min())
            lm = int(100 * Y_predict.max())
            y_min = 10000000            
            theta = -10
            
            for jk in range(ll,lm):
                tht = jk/100.0
                Y_prdct = (Y_predict>=theta).astype(int)
                if abs(sum(Y_prdct)-sum(Y_orig))<y_min:
                #if 1:
                    theta = tht
                    y_min = abs(sum(Y_prdct)-sum(Y_orig))
                    print y_min
                
                    
            Y_predict = (Y_predict>=tht).astype(int)
            Y_predict = np.reshape(Y_predict,[len(Y_predict),1])
            
            #Y_predict = (Y_predict>=theta).astype(int)
            temp = np.multiply((Y_predict==1).astype(int),(Y_orig==1).astype(int))
            opt_score_true_pos = opt_score_true_pos + sum(temp)/(sum(Y_orig==1)+0.0001)
            
            temp = np.multiply((Y_predict==0).astype(int),(Y_orig==0).astype(int))
            opt_score_true_neg = opt_score_true_neg + sum(temp)/(sum(Y_orig==0)+0.0001)
            #opt_score = np.linalg.norm(Y_predict.ravel()-Y_orig.ravel())
            
            mem_pot = np.dot(A_orig,W)
            no_spikes = sum(Y_orig)
            t_fire = np.nonzero(Y_orig)[0]            
            
            peak_inds,peak_values = detect_spike_peaks(mem_pot,no_spikes,t_fire)
            pdb.set_trace()
            pred_spikes = np.zeros(mem_pot.shape)
            pred_spikes[peak_inds] = .5
        #----------------------------------------------------------
    
    opt_score_true_pos = opt_score_true_pos/float(len(T_array))
    opt_score_true_neg = opt_score_true_neg/float(len(T_array))
    
    
    return opt_score_true_pos,opt_score_true_neg



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
                Delta_W,d_alp_vec,t_start,t_end,cst,memory_used  = infer_w_block(W_tot,A,YA,lambda_temp,len_v,0,block_size,inferece_params)
                
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
def delayed_inference_constraints_hinge(out_spikes_tot_mat_file,TT,n,max_itr_opt,sparse_thr_0,alpha0,theta,neuron_range):
    
    #----------------------------Initilizations--------------------------------    
    from auxiliary_functions import soft_threshold
    import os.path
    m = n
    
    
    range_tau = range(0,max_itr_opt)
    
    if len(neuron_range) == 0:
        neuron_range = np.array(range(0,m))
    
    if TT > 20000:
        
        T_temp = 50                              # The size of the initial batch to calculate the initial inverse matrix
        block_size = 1000000        
        T0 = 50 #max(TT - 1*block_size-10,50)                                  # It is the offset, i.e. the time from which on we will consider the firing activity
    else:
        T0 = 0
        T_temp = 1000
        block_size = 500
        
    range_TT = range(T0,TT)
    #--------------------------------------------------------------------------
    
    #-----------------------------Behavior Flags-------------------------------
    der_flag = 0                                # If 1, the derivative criteria will also be taken into account
    rand_sample_flag = 0                        # If 1, the samples will be wide apart to reduce correlation
    sketch_flag = 0                             # If 1, random sketching will be included in the algorithm as well
    load_mtx = 0
    mthd = 2
    #--------------------------------------------------------------------------
    
    #---------------------------Neural Parameters------------------------------
    tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
    tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
    h0 = 0.0                                        # The reset membrane voltage (in mV)
    delta = 0.25                                       # The tanh coefficient to approximate the sign function
    d_max = 10
    t_gap = 2                                    # The gap between samples to consider
    t_avg = 1
    theta = 0
    c_1 = 1                                        # This is the weight of class +1 (i.e. y(t) = 1)
    c_0 = .1                                         # This is the weight of class 0 (i.e. y(t) = 0)
    if theta:
        len_v = n        
    else:
        len_v = n+1
    
    
    W_infer = np.zeros([int(len(range_TT)/float(block_size))+1,len_v])
    W_inferred = np.zeros([len_v,len_v])
    
    t0 = math.log(tau_d/tau_s) /((1/tau_s) - (1/tau_d))
    U0 = 2/(np.exp(-t0/tau_d) - np.exp(-t0/tau_s))  # The spike 'amplitude'
    #--------------------------------------------------------------------------
    
    #---------Identify Incoming Connections to Each Neuron in the List---------
    for ijk in neuron_range:
        
        print '-------------Neuron %s----------' %str(ijk)
    
        #---------------------Necessary Initializations------------------------
        prng = RandomState(int(time()))
        
        lambda_tot = np.zeros([len(range_TT),1])
        no_blocks = 1+len(range_TT)/block_size
        
        W_tot = np.zeros([len_v-1,1])
        Z_tot = np.zeros([len_v-1,1])
        
        dual_gap = np.zeros([len(range_tau),no_blocks])
        total_cost = np.zeros([len(range_tau),1])
        total_Y = np.zeros([len(range_tau),1])
        beta_K = 1
        ell =  block_size
        #----------------------------------------------------------------------
        
        for ttau in range_tau:
            
            #----------------------In-Loop Initializations---------------------
            t_counter = 0
            r_count = 0
            block_count = 0 
            
            alpha = alpha0/float(1+math.log(ttau+1))
            sparse_thr = sparse_thr_0/float(1+math.log(ttau+1))
            itr_W = 0
            
            Delta_Z = np.zeros([len_v-1,1])
            Delta_W = np.zeros([len_v-1,1])
            range_T = range(T0,TT,block_size)
            
            X = np.zeros([len_v,block_size])
            V = np.zeros([len_v,block_size])
            x = np.zeros([len_v,1])
            v = np.zeros([len_v,1])
            xx = np.zeros([len_v,1])
            vv = np.zeros([len_v,1])
            Y = np.zeros([block_size])
            
            
            #------------------------------------------------------------------
            
            for t_0 in range_T:
                
                W_temp = W_tot
                
                #------------This is for the last block-----------
                if (max(range_T)-t_0) < block_size:
                    print 'Oops daisy!'
                    continue
                #-------------------------------------------------
                
                #--------------Check If the Block Is Processed Before-----------
                spikes_file = out_spikes_tot_mat_file[:-4] + '_b_' + str(block_size) + '_c_' + str(t_0) + '_i_' + str(ijk) + '_A.txt'
                if not os.path.isfile(spikes_file):
                    X = np.zeros([len_v,block_size])
                    V = np.zeros([len_v,block_size])
                    AA = np.zeros([len_v,block_size])
                    x = np.zeros([len_v,1])
                    v = np.zeros([len_v,1])
                    xx = np.zeros([len_v,1])
                    vv = np.zeros([len_v,1])
                    Y = np.zeros([block_size])
                    
        
                    yy = 0
                    
                    t_counter = 0
                    t_tot = 0
                    
                    range_temp = range(t_0,t_0+block_size)
                    for t in range_temp:
                        
                        #........Pre-compute Some of Matrices to Speed Up the Process......
                        #fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
                        fire_t = read_spikes_lines(out_spikes_tot_mat_file,t,n)
                        if (ijk in fire_t):                
                            yy = 1
                            x = np.zeros([len_v,1])
                            v = np.zeros([len_v,1])
                        else:
                            yy = 0
                        
                        fire_t = read_spikes_lines(out_spikes_tot_mat_file,t-1,n)
                        fire_t = np.array(fire_t).astype(int)
                        
                        x = math.exp(-1/tau_s) * x
                        try:
                            x[fire_t] = x[fire_t] + 1
                        except:
                            pdb.set_trace()
                        
                        v = math.exp(-1/tau_d) * v
                        v[fire_t] = v[fire_t] + 1
                        
                        if not theta:
                            v[-1,0] = -1.0
                            
                        V[:,t_tot] = v.ravel()
                        X[:,t_tot] = x.ravel()
                        Y[t_tot] = yy
                        
                        #AA[:,t_tot] = (2*yy-1) * v.ravel()
                            
                        t_tot = t_tot + 1
                        
                    Y = np.array(Y)
                    
                    V = V[:,0:t_tot]
                    X = X[:,0:t_tot]
                    Y = Y[0:t_tot]
                    #AA = AA[:,0:t_tot]
                    
                    YY = (Y>0).astype(int) - (Y<=0).astype(int)
                    #A = (V-X).T
                    A = (V).T
                    #AA = AA.T
                    #--------Shift Post-Synaptic Spike One to the Left---------        
                    YY = np.roll(YY,-1)
                    YY[-1] = -1
                    #----------------------------------------------------------

                    #AA = np.dot(np.diag(YY.ravel()),A)
                    AA = np.zeros(A.shape)
                    for t in range(0,t_tot):
                        AA[t,:] = YY[t]*A[t,:]
                        
                    AA = np.delete(AA.T,ijk,0).T
        
                    #------------------Try to Store the Matrix-----------------
                    #pdb.set_trace()
                    file_saved = 0
                    if 0:
                        try:
                            spikes_file_AA = spikes_file
                            np.savetxt(spikes_file,AA,'%2.5f',delimiter='\t')
                            file_saved = 1
                        except:
                            print 'Integrate file could not be saved'
                            file_saved = 0
                            
                        try:
                            spikes_file_YY = out_spikes_tot_mat_file[:-4] + '_b_' + str(block_size) + '_c_' + str(t_0) + '_i_' + str(ijk) + '_Y.txt'
                            np.savetxt(spikes_file_YY,YY,'%2.1f',delimiter='\t')
                            file_saved = 1
                        except:
                            file_saved = 0
                            print 'Spikes file could not be saved'
                    #----------------------------------------------------------
                else:
                    
                    print 'yoohoo!'
                    found_flag = 1                    
                    spikes_file_AA = out_spikes_tot_mat_file[:-4] + '_b_' + str(block_size) + '_c_' + str(t_0) + '_i_' + str(ijk) + '_A.txt'
                    spikes_file_YY = out_spikes_tot_mat_file[:-4] + '_b_' + str(block_size) + '_c_' + str(t_0) + '_i_' + str(ijk) + '_Y.txt'
                    file_saved = 1
                    
                    if load_mtx:
                        AA = np.genfromtxt(spikes_file_AA, dtype=float, delimiter='\t')
                        YY = np.genfromtxt(spikes_file_YY, dtype=float, delimiter='\t')
                        #pdb.set_trace()
                #---------------------------------------------------------------
                
                if load_mtx:
                    #-------------Add a Row for Theta If Missing From File----------
                    if AA.shape[1] != len(W_temp):
                        tmp = -YY
                        tmp = np.reshape(tmp,[len(tmp),1])
                        pdb.set_trace()
                        AA = np.hstack([AA,tmp])
                    #---------------------------------------------------------------
                    
                    #------------------Prepare the Submatrix-----------------------
                    if rand_sample_flag:
                        t_init = np.random.randint(0,t_gap)
                        t_inds = np.array(range(t_init,block_size,t_gap))
                        
                        aa = AA[t_inds,:]
                        yy = YY[t_inds]
                    else:
                        aa = AA
                        yy = YY
                    #---------------------------------------------------------------
                
                #--------------Assign Weights to the Classes--------------------                
                gg = {}
                gg[-1] = c_0
                gg[1] = c_1
                #GG = np.diag(gg)
                #---------------------------------------------------------------
                
                #-----------------------Do the Optimization---------------------
                if load_mtx:
                    TcT = len(yy)
                else:
                    TcT = block_size
                lamb = .00001/float(TcT)
                cf = lamb*TcT
                ccf = 1/float(cf)
                
                #pdb.set_trace()        
                lambda_temp = lambda_tot[block_count*block_size:(block_count+1)*block_size]
                
                if rand_sample_flag:
                    lambda_0 = lambda_temp[t_inds]
                else:
                    lambda_0 = lambda_temp
                    
                d_alp_vec = np.zeros([block_size,1])
                
                if 0:
                    qq = np.ones([TcT,2])
                    qq[:,0] = 0
                    qq[:,1] = 1
                    bns = list(qq)
                    bb = c_1 * (yy>0).astype(int) + c_0 * (yy<=0).astype(int)
                    #bb = np.dot(np.reshape(bb,[len(bb),1]),np.ones([1,len_v-1]))
                    #bb = np.multiply(bb,aa)
                    #bb = np.diag(bb.ravel())
                    #bb = np.dot(bb,aa)
                    #pdb.set_trace()
                
                    bb = aa
                    
                    
                    cb = np.ones([TcT,1]) - 1 * np.dot(bb,W_tot) 
                    opt = {'disp':False,'maxiter':25000}
                    FF = bb.T
                #res_cons = optimize.minimize(hinge_loss_func_dual, lambda_0, args=(FF,cb,0.5/cf),jac=hinge_jac_dual,bounds=bns,constraints=(),method='L-BFGS-B', options=opt)
                #FF = np.dot(bb,bb.T)
                
                #res_cons = optimize.minimize(hinge_loss_func_dual_l2, lambda_0, args=(FF,cb,0.5/cf),jac=hinge_jac_dual_l2,bounds=bns,constraints=(),method='L-BFGS-B', options=opt)
                #print res_cons['message']
                #lam = np.reshape(res_cons['x'],[TcT,1])
                #lam = np.multiply(lam,(lam > qq.min()).astype(int))
                #d_alp_vec[t_inds] = lam
                cst = 0
                cst_y = 0
                cst_old = 0
                for ss in range(0,2*TcT):
                            
                    ii = np.random.randint(0,TcT)
                    if rand_sample_flag:
                        jj = t_inds[ii]
                    else:
                        jj = ii
                            
                    #~~~~~~~~~~~Find the Optimal Delta-Alpha~~~~~~~~~~~
                    #aa_t = read_spikes_lines_integrated(spikes_file_AA,ii+1,n)
                    #yy_t = read_spikes_lines_integrated(spikes_file_YY,ii+1,1)
                    #yy_t = yy_t[0]
                    if load_mtx:
                        aa_t = aa[ii,:]                        
                        yy_t = yy[ii]
                    else:
                        if file_saved:
                            aa_t = read_spikes_lines_integrated(spikes_file_AA,ii+1,n)
                            yy_t = read_spikes_lines_integrated(spikes_file_YY,ii+1,1)
                            yy_t = yy_t[0]
                        else:
                            if not rand_sample_flag:
                                aa_t = AA[ii,:]
                                yy_t = YY[ii]
                        
                    
                    try:
                        ff = gg[yy_t]*(aa_t)
                    except:
                        pdb.set_trace()
                    
                    if theta:
                        c = 1 + theta * yy_t
                    else:
                        c = 1
                    
                    
                    
                    #~~~~~~~~~~~~Method 2~~~~~~~~~~~~~
                    if mthd == 2:
                        b = (c-np.dot(W_temp.T,aa_t))/(0.00001+pow(np.linalg.norm(aa_t),2))
                        b = min(ccf-lambda_temp[jj],b)
                        b = max(-lambda_temp[jj],b)
                        d_alp = b
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~Method 3: Sparsity~~~~~~~~~~~~~
                    elif mthd == 3:
                        
                        
                        aac = np.ones([1,2])
                        aac[:,0] = -lambda_temp[jj]
                        aac[:,1] = ccf-lambda_temp[jj]
                        bns = list(aac)
                        
                        res_cons = optimize.minimize(l1_loss,0, args=(W_temp,ff),bounds=bns,constraints=(),method='TNC', options={'disp':False,'maxiter':500})
                        
                        b = res_cons['x']
                        
                        d_alp = b[0]
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    else:
                        b = cf * (np.dot(W_temp.T,ff) - c)/pow(np.linalg.norm(aa_t),2)
                        
                        try:
                            if (b<=lambda_temp[jj]+1) and (b >= lambda_temp[jj]-1):
                                d_alp = -b
                            elif pow(b-lambda_temp[jj],2) < pow(b+1-lambda_temp[jj],2):
                                d_alp = -1-lambda_temp[jj]
                            else:
                                d_alp = 1-lambda_temp[jj]
                        except:
                            pdb.set_trace()
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            
                    lambda_temp[jj] = lambda_temp[jj] + d_alp
                    d_alp_vec[jj] = d_alp_vec[jj] + d_alp
                    #W_temp = W_temp + d_alp * np.reshape(aa[ii,:],[len_v-1,1])/float(cf)
                    if (mthd == 2) or (mthd == 3):
                        W_temp = W_temp + d_alp * np.reshape(aa_t,[len_v-1,1])
                    elif mthd == 1:
                        W_temp = W_temp + d_alp * np.reshape(aa_t,[len_v-1,1])/float(cf)
                    else:
                        xx = np.dot(W_temp.T,aa_t)
                        W_temp = W_temp + 0.001*abs(gg[yy_t]) * (np.reshape(aa_t,[n,1]) * 0.5 * (np.sign(xx-1) + np.sign(xx-10))) 
                    
                    
                    cst = cst + np.sign(hinge_loss_func(W_temp,-aa_t,.1,1,0))
                    if yy_t:
                        cst_y = cst_y + hinge_loss_func(W_temp,-aa_t,0.1,1,0)
                        
                    if not (ss%1000000):
                        
                        print mthd,cst - cst_old
                        cst_old = cst
                #---------------------------------------------------------------
            
                #----------------------Update the Weights-----------------------
                #pdb.set_trace()
                #Delta_W_loc = np.dot(bb.T,d_alp_vec[t_inds])
                if mthd == 4:
                    Delta_W_loc = W_temp
                    #Delta_W_loc = soft_threshold(W_temp.ravel(),sparse_thr)
                else:
                    Delta_W_loc = W_temp
                    if 0:
                        if rand_sample_flag:
                            Delta_W_loc = np.dot(aa.T,d_alp_vec[t_inds])
                        else:
                            Delta_W_loc = np.dot(aa.T,d_alp_vec)
                Delta_W = Delta_W + Delta_W_loc
                W_tot = W_tot + np.reshape(Delta_W_loc,[len_v-1,1])/no_blocks
                lambda_tot[block_count*block_size:(block_count+1)*block_size] = lambda_tot[block_count*block_size:(block_count+1)*block_size] + d_alp_vec * (beta_K/no_blocks)
                #---------------------------------------------------------------

                #------------------Evaluate the Performance---------------------
                #BB = np.dot(0*np.eye(TcT) + theta * np.diag(YY.ravel()),np.ones([TcT,1]))
                if 0:
                    BB = 0*np.ones([TcT,1])
                    print hinge_loss_func(Delta_W_loc,-aa,BB,1,0)
                        
                block_count = block_count + 1
                #----------------------------------------------------------
                
                #----------------------Calculate Cost----------------------
                if 0:
                    cst = np.dot(AA,Delta_W)
                    
                    bb = 0*YY           # 0 corresponds to theta since we want to learn it as well
                
                    total_cost[ttau] = total_cost[ttau] + sum(cst.ravel()<bb)
                    total_Y[ttau] = total_Y[ttau] + sum(np.multiply(YY>0,(cst.ravel()<bb).ravel()))
                    Yk = (YY>0).astype(int)
                else:
                    total_cost[ttau] = total_cost[ttau] + cst
                    total_Y[ttau] = total_Y[ttau] + cst_y
                #DD = np.dot(np.diag(YY),AA)
                #DD[:,-1] = -np.ones([block_size])
                #
                #cst = np.dot(DD,2*W_tot)
                
                
                #cst = np.dot(DD,2*Delta_W_loc)
                #total_cost[ttau] = total_cost[ttau] + sum(np.sign(cst.ravel())!=np.sign(YY))
                
                #cc = np.dot(DD,2*W_tot)
                #total_Y[ttau] = total_Y[ttau] + sum(Y_orig>0)
                
                #total_Y[ttau] = total_Y[ttau] + sum(np.multiply(YY>0,np.sign(cst.ravel())!=np.sign(YY)))
                
                #total_Y[ttau] = total_Y[ttau] + sum(np.multiply(YY<=0,(cst<0).ravel()))
                #----------------------------------------------------------
                    
                #..................................................................
            
                
            #pdb.set_trace()
            #W_tot = W_tot + Delta_W/no_blocks
            st_cof = 0.1/float(1+ttau)
            #W_tot = W_tot + Delta_W/no_blocks
            
            WW = np.zeros([len_v,1])
            WW[0:ijk,0] = W_tot[0:ijk,0]
            WW[ijk+1:,0] = W_tot[ijk:,0]
            
            
            print total_cost[ttau],total_Y[ttau]
            #pdb.set_trace()
            if ttau > 0:
                if ((total_cost[ttau] == 0) and (total_cost[ttau-1] == 0)) or (total_cost[ttau] - total_cost[ttau-1] == 0):
                    #pdb.set_trace()
                    break
            if not ((ttau+1) % 8):
                #W2 = merge_W(W_infer[0:itr_W,:],0.01)
                print total_cost[0:ttau]
                #DD = np.dot(np.diag(YY),AA)
                #cc = np.dot(DD,2*W_tot)
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
            
            #Y_predict = np.dot(A,W)
            Y_predict = np.dot(A_orig,W)
            Y_predict = (Y_predict>0).astype(int)
            #np.linalg.norm(Y_predict-Y_orig)
            opt_score = np.linalg.norm(Y_predict.ravel()-Y_orig)
            pdb.set_trace()
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #pdb.set_trace()
        W_inferred[0:len_v,ijk] = WW[0:len_v].ravel()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            
    return W_inferred



#------------------------------------------------------------------------------

