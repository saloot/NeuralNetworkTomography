#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
#from brian import *
from scipy import sparse
import numpy as np
import pdb,os,sys
import random
import copy
import numpy.ma as ma
from default_values import *
#==============================================================================



#==============================================================================
#===========================Determine_Binary_Threshold=========================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function determines the two thresholds for deciding if a connection is
# excitatory or inhibitory. It uses probabilistic arguments or clustering
# approaches to determine the thresholds, depending the method chosen.

# INPUTS:
#    method: 'c' or 'p', for clustering-base or probabailistic way of categorizing connections
#    params: for the 'p' method, specifies probability of having excitatory or inhibitory connections.
#            for the 'c' method, specifies the number of classes (usually 3) and the adjustment factors for each class
#    obs:    the vector of the weights for the incoming connections.  

# OUTPUTS:
#    thr_inh:  the threshold for classifying connections as inhibitory
#    thr_zero: the threshold for classifying connections as "void" or non-existent
#    thr_exc:  the threshold for classifying connections as excitatory
#------------------------------------------------------------------------------

def determine_binary_threshold(method,params,obs):
    
    #---------------------Import Necessary Libraries---------------------------
    from math import sqrt
    import numpy as np
    from scipy.cluster.vq import kmeans 
    #--------------------------------------------------------------------------
    
    #--------------------------The Probablistic Method-------------------------
    if (method == 'p'):
        
        #.....................Get the Simulation Parameters....................
        n = len(obs)
        p_exc = params[0]
        p_inh = params[1]
        #......................................................................
        
        #.....................Calculate the Thresholds.........................
        gamma_pos_range = np.array(range(0,int(1000*obs.max())))/1000.0
        gamma_neg_range = -np.array(range(0,abs(int(1000*obs.min()))))/1000.0
        
        itr_e = 0
        for gamma_pos in gamma_pos_range:
            W = (obs>gamma_pos).astype(int)
            if sum(W)/float(n) < p_exc:
                break
            itr_e = itr_e + 1
        
        if (itr_e < len(gamma_pos_range)):
            thr_exc = gamma_pos_range[itr_e]            
        else:
            thr_exc = float('nan')
        
        itr_e = 0        
        for gamma_neg in gamma_neg_range:
            W = (obs < gamma_neg).astype(int)
            if sum(W)/float(n) < p_inh:
                break
            itr_e = itr_e + 1
        
        if (itr_e < len(gamma_neg_range)):
            thr_inh = gamma_neg_range[itr_e]
        else:
            thr_inh = float('nan')
        #......................................................................
        
    #--------------------------------------------------------------------------
    
    #---------------------The Clustering Based Approach------------------------
    elif (method == 'c'):
        
        #.....................Get the Simulation Parameters....................        
        n = len(obs)
        adj_factor_exc = params[0]
        adj_factor_inh = params[1]
        no_clusters = params[2]
        #......................................................................
        
        #.........Computhe the Thresholds Using the K-Means Algorithms.........
        success_itr = 0
        centroids = np.zeros([no_clusters])
        res = 0
        temp= np.zeros([no_clusters])
        while success_itr<10:
            temp,res = kmeans(obs,no_clusters,iter=30)
            success_itr = success_itr + 1
            if len(temp) == len(centroids):
                centroids = centroids + temp
            else:
                if len(temp) < len(centroids):
                    centroids = centroids + np.hstack([temp,np.zeros(len(centroids)-len(temp))])
                else:
                    centroids = centroids + temp[0:len(centroids)]
        
        centroids = np.divide(centroids,float(success_itr))
        
        ss = np.sort(centroids)
        val_inh = ss[0]
        val_exc = ss[2]
        thr_zero = ss[1]
        #......................................................................
        
        #.......................Adjust the Thresholds..........................
        min_val = np.min(obs)# - (thr_zero-np.min(obs))-0.01
        max_val = np.max(obs)# - (thr_zero-np.max(obs))+0.01
        
        #thr_inh = val_inh + (adj_factor_inh -1)*(val_inh - min_val)
        thr_inh = val_inh + (adj_factor_inh -1)*(val_inh - thr_zero)
        thr_inh = np.min([thr_inh,thr_zero-.01])
        thr_inh = np.max([thr_inh,min_val+.01])
        
        #thr_exc = val_exc + (adj_factor_exc -1)*(val_exc - max_val)
        thr_exc = val_exc + (adj_factor_exc -1)*(val_exc - thr_zero)
        thr_exc = np.max([thr_exc,thr_zero+.01])
        thr_exc = np.min([thr_exc,max_val-.01])
            
        if no_clusters > 3:
            thr_exc2 = thr_zero_r - (0.51)*(thr_zero_r-thr_exc)
            thr_exc2 = np.max([thr_exc2,thr_exc+.01])
            thr_exc = thr_exc2
        else:
            thr_exc2 = None
            
        #......................................................................
        
        
    #--------------------------------------------------------------------------
    
    
    return [thr_inh,thr_zero,thr_exc]
#==============================================================================
#==============================================================================        



#==============================================================================
#=============================beliefs_to_ternary================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function transforms the beliefs matrix to a binary matrix, with +1,-1,
# and 0 entries.

# INPUTS:
#    ternary_mode: '4' for clustering-based approach (using K-Means)
#                  '2' for sorting-based approach
#                  '7' for the conservative approach of only assigning those edges that we are sure about (far from mean values)
#    W_inferred: The 'association-matrix' for the inferred graph
#    params: specifies the necessary parameters for each ternarification algorithm
#    dale_law_flag: if '1', the Dale law will be enforced and all outgoing connections for a neuron will have the same type

# OUTPUTS:
#    W_binary: the ternary adjacency matrix
#    centroids: the class represntative for the clustering-based appraoch (using K-Means)
#------------------------------------------------------------------------------

def beliefs_to_ternary(ternary_mode,W_inferred,params,dale_law_flag):
    
    #---------------------Import Necessary Libraries---------------------------
    import numpy as np
    from scipy.cluster.vq import kmeans,whiten,kmeans2,vq
    #--------------------------------------------------------------------------
    
    #--------------------------Initializing Variables--------------------------
    centroids = []
    
    a = W_inferred.shape
    lll = len(a)
    
    if lll>1:
        n = a[0]
        m = a[1]
    elif lll == 1:
        n = a[0]
        m = 1
    else:
        n = 1
        m = 0
    
    if m:
        W_inferred = W_inferred.reshape(n,m)   
    
    if min(a) == 1:
        lll = 1
    #--------------------------------------------------------------------------
    
    #---------Determine the Type of Network: FeedForward or Recurrent----------
    W_binary = np.zeros([n,m])
    if ( (ternary_mode == 4) or (ternary_mode == 5)):
        centroids = np.zeros([m,3])
    W_binary.fill(0)
    #--------------------------------------------------------------------------
    
    
    #----------------Binary Mode 2: Sorting-Based Thresholding-----------------
    if (ternary_mode == 2):
        
        #..................Get the Binarification Parameters...................
        p_exc = params[0]
        p_inh = params[1]
        #......................................................................
        
        #............Identify Incoming Connections for Each Neuron.............
        for i in range(0,m):
                
                thr_inh,thr_zero,thr_exc = determine_binary_threshold('p',params,W_inferred[:,i])
                W_binary[:,i] = (W_inferred[:,i] > thr_exc).astype(int) - (W_inferred[:,i] < thr_inh).astype(int)
        #......................................................................
        
    #--------------------------------------------------------------------------
    
    
    #--------------Binary Mode 4: Clustering-Based Thresholding----------------
    elif (ternary_mode == 4):

        #......................Determine the Thresholds........................
        fixed_inds = params[2]
        for i in range(0,m):            
            W_inferred_temp = copy.deepcopy(W_inferred[:,i])
            
            #~~~~~~~~~~Take Out All Fixed Entries from Classification~~~~~~~~~~
            mask_inds = fixed_inds
            mask_inds[i,i] = 1
            temp = np.ma.masked_array(W_inferred_temp,mask= mask_inds[:,i])
            masked_inds = np.nonzero((temp.mask).astype(int))
            masked_vals = temp.compressed()
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #~---~~~~~~~~~~~~~Classify Incoming Edges~~~~~~~~~~~~~~~~~~~~~~~~~~
            params = params[0:2]
            params.append(3)                        # "3" refers to the number of classes, namely, excitatory, inhibitory and non-existent
            if (sum(abs(masked_vals))):
                thr_inh,thr_zero,thr_exc = determine_binary_threshold('c',params,masked_vals)
                centroids[i,:] = [thr_inh,thr_zero,thr_exc]
                if sum(abs(centroids[i,:])):
                    W_temp,res = vq(masked_vals,np.array([thr_inh,thr_zero,thr_exc]))
                    W_temp = W_temp - 1
                else:
                    W_temp = np.zeros(masked_vals.shape)    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            
            #~~~~~~~~~~~~~~~Role Back Values to Unmasked Indices~~~~~~~~~~~~~~~
            mask_counter = 0
            for j in range(0,n):                
                if j in masked_inds[0]:                    
                    W_binary[j,i] = np.sign(W_inferred[j,i])
                else:
                    W_binary[j,i] = W_temp[mask_counter]
                    mask_counter = mask_counter + 1
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        #......................................................................
        
    #--------------------------------------------------------------------------
    
    
    #-----Binary Mode 7: Only Assigning Those Edges That We Are Sure About-----
    elif (ternary_mode == 7):
        
        # We are going to select edges that are far from mean        
        fixed_inds = params[0]
        a = params[1]
        b = params[2]
        W_binary = nan * np.ones([n,m])
        
        for i in range(0,m):
            W_inferred_temp = copy.deepcopy(W_inferred[:,i])            
            temp = np.ma.masked_array(W_inferred_temp,mask= fixed_inds[:,i])
            masked_inds = np.nonzero((temp.mask).astype(int))
            masked_vals = temp.compressed()
            
            W_temp = float('nan')*np.ones([len(masked_vals),1])
            
            if sum(sum(abs(masked_vals))):
                max_val = float(masked_vals.max())
                mean_val = masked_vals.mean()
                min_val = float(masked_vals.min())
                var_val = pow(masked_vals.var(),0.5)
            
                #----------------------Assign Excitatory Edges--------------------------
                temp = (masked_vals > mean_val + a*var_val).astype(int)
                exc_ind = np.nonzero(temp)
                W_temp[exc_ind,0] = 0.001
                #------------------------------------------------------------------
            
                #-------------------Assign Inhibitory Edges------------------------            
                temp = (masked_vals < mean_val - b*var_val).astype(int)
                inh_ind = np.nonzero(temp)
                W_temp[inh_ind,0] = -0.005
                #------------------------------------------------------------------
                        
                #-------------------------Assign Void Edges------------------------
                temp = (masked_vals > mean_val - 0.1*var_val).astype(int)
                temp = np.multiply(temp,(masked_vals < mean_val + 0.05*var_val).astype(int))
                void_ind = np.nonzero(temp)
                W_temp[void_ind,0] = 0.0
                #------------------------------------------------------------------
            
            
                #----------------Role Back Values to Unmasked Indices--------------
                mask_counter = 0
                for j in range(0,n):                
                    if j in masked_inds[0]:
                        W_binary[j,i] = W_inferred[j,i]
                    else:
                        W_binary[j,i] = W_temp[mask_counter,0]
                        mask_counter = mask_counter + 1
                #------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    
    #----If Necessary, Set All Outgoing Connections of a Neuron to one Type----
    if dale_law_flag:
        for i in range(0,n):
            w_temp = W_binary[i,:]
            d = sum(w_temp)
            w_temp_s = np.sort(W_inferred[i,:])
            ind = np.argsort(w_temp)
            w_temp = np.zeros(n)
            if (d>2):
                w_temp[ind[len(ind)-int(round(connection_prob*n))+1:len(ind)+1]] = 1
                W_binary[i,:] = w_temp
                
            elif (d< -1):
                w_temp[ind[0:int(round(connection_prob*n))+1]] = -1
                W_binary[i,:] = w_temp
    #--------------------------------------------------------------------------    
        
    return W_binary,centroids
#==============================================================================
#==============================================================================


#==============================================================================
#=============================caculate_accuracy================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function compares the inferred binary matrix and compare it with the
# original graph to calculate the accruacy (recall and precision) of the
# algorithm

# INPUT:
#    W_binary: the inferred ternary graph
#    W: the actual graph (ground truth)

# OUTPUT:
#    recall:    the recall for inhibitory, non-existent and excitatory connections
#    precision: the precision for inhibitory, non-existent and excitatory connections
#------------------------------------------------------------------------------

def caculate_accuracy(W_binary,W):
    
    #--------------------------Initializing Variables--------------------------
    a = W.shape    
    n = a[0]
    lll = len(a)
    #--------------------------------------------------------------------------
    
    #----------------Compute Accuracy for Recurrent Networks-------------------
    if ( (lll>1) and (min(a)>1)):
        
        A = np.zeros(W.shape)
        if (sum(sum(W>A))):
            acc_plus = float(sum(sum(np.multiply(W_binary>A,W>A))))/float(sum(sum(W>A)))
        else:
            acc_plus = float('NaN')
        
        if (sum(sum(W<A))):
            acc_minus = float(sum(sum(np.multiply(W_binary<A,W<A))))/float(sum(sum(W<A)))
        else:
            acc_minus = float('NaN')
        
        if (sum(sum(W==A))):
            acc_zero = float(sum(sum(np.multiply(W_binary==A,W==A))))/float(sum(sum(W==A)))
        else:
            acc_zero = float('NaN')
        
        if (sum(sum(W_binary>A))):
            prec_plus = float(sum(sum(np.multiply(W_binary>A,W>A))))/float(sum(sum(W_binary>A)))
        else:
            prec_plus = float('NaN')
        
        if (sum(sum(W_binary<A))):
            prec_minus = float(sum(sum(np.multiply(W_binary<A,W<A))))/float(sum(sum(W_binary<A)))
        else:
            prec_minus = float('NaN')
            
        if (sum(sum(W_binary==A))):
            prec_zero = float(sum(sum(np.multiply(W_binary==A,W==A))))/float(sum(sum(W_binary==A)))
        else:
            prec_zero = float('NaN')
    #--------------------------------------------------------------------------
    
    #---------------Compute Accuracy for FeedForward Networks------------------
    else:
        A = np.zeros([n,1])
        W_binary = W_binary.reshape([n,1])
        if (sum(W>A)):
            acc_plus = float(sum(np.multiply(W_binary>A,W>A)))/float(sum(W>A))
        else:
            acc_plus = float('NaN')
        if (sum(W<A)):
            acc_minus = float(sum(np.multiply(W_binary<A,W<A)))/float(sum(W<A))
        else:
            acc_minus = float('NaN')
        if (sum(W==A)):
            acc_zero = float(sum(np.multiply(W_binary==A,W==A)))/float(sum(W==A))
        else:
            acc_zero = float('NaN')

        if (sum(W_binary>A)):
            prec_plus = float(sum(np.multiply(W_binary>A,W>A)))/float(sum(W_binary>A))
        else:
            prec_plus = float('NaN')
    
        if (sum(W_binary<A)):
            prec_minus = float(sum(np.multiply(W_binary<A,W<A)))/float(sum(W_binary<A))
        else:
            prec_minus = float('NaN')

        if (sum(W_binary==A)):
            prec_zero = float(sum(np.multiply(W_binary==A,W==A)))/float(sum(W_binary==A))
        else:
            prec_zero = float('NaN')
        
        
    #--------------------------------------------------------------------------
    
    #---------------------Reshape and Return the Results-----------------------
    recall = [acc_plus,acc_minus,acc_zero]
    precision = [prec_plus,prec_minus,prec_zero]
    return recall,precision
    #--------------------------------------------------------------------------
#==============================================================================    


#==============================================================================
#========================parse_commands_ternary_algo===========================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function runs the neural networks and generatethe required neural
# activity. The Brian simulator is used for this part.

# INPUTS:
#    input_opts: the options provided by the user
#------------------------------------------------------------------------------

def parse_commands_ternary_algo(input_opts):
    if (input_opts):
        for opt, arg in input_opts:
            if opt == '-Q':
                frac_stimulated_neurons = float(arg)                # Fraction of neurons in the input layer that will be excited by a stimulus
            elif opt == '-T':
                T_max = int(arg)                         # Number of times we inject stimulus to the network
            elif opt == '-S':
                ensemble_size = int(arg)                            # The number of random networks that will be generated                
            elif opt == '-A':
                file_name_base_data = str(arg)                      # The folder to store results
            elif opt == '-F':
                ensemble_count_init = int(arg)                      # The ensemble to start simulations from        
            elif opt == '-R':
                random_delay_flag = int(arg)                        # The ensemble to start simulations from            
            elif opt == '-B':
                ternary_mode = int(arg)                              # Defines the method to transform the graph to binary. "1" for threshold base and "2" for sparsity based                        
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
            elif opt == '-J': 
                delay_known_flag = str(arg)                         # If 'Y', we assume that the delay is known during the inference algorithm
            elif opt == '-U': 
                beta = int(arg)                                     # Specify the update probability paramter (p = 1/beta) in STOCHASTIC NEUINF
            elif opt == '-Z': 
                alpha0 = float(arg)                                 # Specify the update learnining rate
            elif opt == '-O': 
                temp = (arg).split(',')                             # The range of recorded durations (T_range)
                T_range = []
                for i in temp:                        
                    T_range.append(int(i))
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
        
    if 'T_max' not in locals():        
        T_max = T_MAX_DEFAULT
        print('ATTENTION: The default value of %s for T_max is considered.\n' %str(T_max))

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
    
    if 'beta' not in locals():
        beta = BETA_DEFAULT
        print('ATTENTION: The default value of %s for beta is considered.\n' %str(beta))
    
    if 'alpha0' not in locals():
        alpha0 = ALPHA0_DEFAULT
        print('ATTENTION: The default value of %s for alpha0 is considered.\n' %str(alpha0))    
    #------------------------------------------------------------------------------
    
    #------------------Create the Necessary Directories if Necessary---------------
    if not os.path.isdir(file_name_base_results):
        os.makedirs(file_name_base_results)    
    if not os.path.isdir(file_name_base_results+'/Inferred_Graphs'):
        temp = file_name_base_results + '/Inferred_Graphs'
        os.makedirs(temp)
    if not os.path.isdir(file_name_base_results+'/Plot_Results'):    
        temp = file_name_base_results + '/Plot_Results'
        os.makedirs(temp)    
    #------------------------------------------------------------------------------


    return frac_stimulated_neurons,T_max,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,ternary_mode,file_name_base_results,inference_method,sparsity_flag,we_know_topology,beta,alpha0,infer_itr_max,T_range


