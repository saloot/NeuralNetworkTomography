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
#===============================Find_Edge_Index================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function finds the layer corresponding to the edge that we speculate
# being excitatory or inhibitory
#------------------------------------------------------------------------------

def find_edge_index(W_estimated,D_estimated,Network,mode):
    n_array = Network.n_exc_array + Network.n_inh_array
    if mode == 'E':
        i,j = np.unravel_index(W_estimated.argmax(),W_estimated.shape)
    else:
        aa = abs(W_estimated)
        i,j = np.unravel_index(aa.argmin(),W_estimated.shape)
    ssum = 0
    lay_i = -1
    lay_j = -1
    for l in range(0,len(n_array)):
        ssum = ssum + n_array[l]
        if (i<ssum) and (lay_i<0):
            lay_i = l
            index_i = i-(ssum-n_array[l])
            #pdb.set_trace()
        if (j<ssum) and (lay_j < 0):
            lay_j = l
            index_j = j-(ssum-n_array[l])
            #pdb.set_trace()
        if (lay_i > 0) and (lay_j >0):
            break
        
    ind = str(lay_i) + str(lay_j)
    temp_list = Network.Neural_Connections[ind]
    W = temp_list[0]
    D = temp_list[1]
    w = W[index_i,index_j]
    d = D[index_i,index_j]
    W_estimated[i,j] = 0
    return ind,w,d,D_estimated[i,j],W_estimated
#==============================================================================        
#==============================================================================



#==============================================================================
#===========================Determine_Binary_Threshold=========================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function determines the two thresholds for deciding if a connection is
# excitatory or inhibitory. It uses probabilistic arguments or clustering
# approaches to determine the thresholds, depending the method chosen
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
        p_exc = params[0]
        p_inh = params[1]
        theta = params[2]
        T = params[3]
        Delta = params[4]
        q = params[5]
        n = params[7]
        #......................................................................
        
        #................Compute the Random Variables Parameters...............
        no_input_ones = q * n
        mu = (p_exc - p_inh) * no_input_ones
        var = math.sqrt((p_exc * (1 - p_exc) - p_inh * (1-p_inh) + 2 * p_exc * p_inh) * no_input_ones)
        mu_p = (p_exc - p_inh) * (no_input_ones - 1)
        var_p = math.sqrt((p_exc * (1 - p_exc) - p_inh * (1-p_inh) + 2 * p_exc * p_inh) * (no_input_ones - 1) )
        #......................................................................
        
        #.............Determine the Probability of Beleif Updates..............
        p_W_plus_1 = q_func_scalar((theta - mu_p - 1)/var_p)                    # Probability that the post-synaptic neuron has fired when a pre-synaptic neuron has fired and there is an excitatory connection between them
        p_W_zero_1 = q_func_scalar((theta - mu_p)/var_p)                        # Probability that the post-synaptic neuron has fired when a pre-synaptic neuron has fired and there is no connection between them
        p_W_minus_1 = q_func_scalar((theta - mu_p + 1)/var_p)                   # Probability that the post-synaptic neuron has fired when a pre-synaptic neuron has fired and there is an inhibitory connection between them
        p_W_plus_0 = 1 - p_W_plus_1                                             # Probability that the post-synaptic neuron has fired when a pre-synaptic neuron has not fired and there is an excitatory connection between them
        p_W_zero_0 = 1 - p_W_zero_1                                             # Probability that the post-synaptic neuron has fired when a pre-synaptic neuron has not fired and there is no connection between them
        p_W_minus_0 = 1 - p_W_minus_1                                           # Probability that the post-synaptic neuron has fired when a pre-synaptic neuron has not fired and there is an inhibitory connection between them
        #......................................................................
        
        #.....................Calculate the Thresholds.........................
        e1 = p_W_plus_1 - p_W_plus_0
        e2 = p_W_minus_1 - p_W_minus_0
        thr_exc_base = e1# - (e1-e2)/4
        thr_inh_base = e2# + (e1-e2)/4
        thr_exc = thr_exc_base * Delta * T * q
        thr_inh = thr_inh_base * Delta * T * q
        thr_zero = 0
        #......................................................................
        
    #--------------------------------------------------------------------------
    
    #---------------------The Clustering Based Approach------------------------
    elif (method == 'c'):
        
        #.....................Get the Simulation Parameters....................        
        n = len(obs)
        adj_factor_exc = params[0]
        adj_factor_inh = params[1]
        #......................................................................
        
        #.........Computhe the Thresholds Using the K-Means Algorithms.........
        success_itr = 0
        centroids = np.zeros([1,3])
        temp2= np.zeros([1,3])
        while success_itr<1:
            temp,res = kmeans(obs,3,iter=30)            
            if 1:
                success_itr = success_itr + 1
                
                #centroids = centroids + temp
                centroids = temp
                
                
        
        centroids = centroids/float(success_itr)
        ss = np.sort(centroids)
        #pdb.set_trace()
        
        #ss = ss[0]
        if (len(ss) == 3):
            val_inh = ss[0]
            val_exc = ss[2]
            thr_zero = ss[1]
        else:
            val_inh = min(ss)
            val_exc = max(ss)
            thr_zero = 0.5 * (val_inh+val_exc)
        #......................................................................
        
        #.......................Adjust the Thresholds..........................
        min_val = np.min(obs) - (thr_zero-np.min(obs))-0.01
        max_val = np.max(obs) - (thr_zero-np.max(obs))+0.01
    
        thr_inh = val_inh + (adj_factor_inh -1)*(val_inh - min_val)
        thr_inh = np.min([thr_inh,thr_zero-.01])
    
        thr_exc = val_exc + (adj_factor_exc -1)*(val_exc - max_val)
        thr_exc = np.max([thr_exc,thr_zero+.01])
        #......................................................................
        
    #--------------------------------------------------------------------------
    
    #-------------The Clustering Based Approach Into Two Groups----------------
    elif (method == 'c2'):
        
        #.....................Get the Simulation Parameters....................        
        n = len(obs)
        adj_factor_exc = params[0]        
        #......................................................................
        
        #.........Computhe the Thresholds Using the K-Means Algorithms.........
        success_itr = 0
        centroids = np.zeros([1,2])
        temp2= np.zeros([1,2])
        while success_itr<1:
            temp,res = kmeans(obs,2,iter=30)            
            if 1:
                success_itr = success_itr + 1                
                centroids = temp
                
                
        
        centroids = centroids/float(success_itr)
        ss = np.sort(centroids)
        
        if (len(ss) == 2):
            val_minus = ss[0]
            val_plus = ss[1]            
        else:
            val_minus = min(ss)
            val_plus = max(ss)            
        #......................................................................
        
        #.......................Adjust the Thresholds..........................
        min_val = np.min(obs) 
        max_val = np.max(obs)
    
        thr_minus = val_minus #+ (adj_factor_inh -1)*(val_inh - min_val)
        #thr_inh = np.min([thr_inh,thr_zero-.01])
    
        thr_plus = val_plus #+ (adj_factor_exc -1)*(val_exc - max_val)
        #thr_exc = np.max([thr_exc,thr_zero+.01])
        #......................................................................
        
    #--------------------------------------------------------------------------
    
    return [thr_inh,thr_zero,thr_exc]
#==============================================================================    
#==============================================================================        


#==============================================================================
#=============================initial_stimulation==============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function randomly chooses a subset of neurons and stimulate them in the
# beginning of the simulations
#------------------------------------------------------------------------------

def initial_stimulation(t):
    rates=np.zeros([n])*Hz    
    
    if t in stimul_times:        
        #input_index = floor(n*rand(round(n*qqi)))
        #input_index = input_index.astype(int)        
        input_index = [neuron_indices[i] for i, x in enumerate(stimul_times) if x == t]
        rates[input_index]=ones(round(n*qqi))*input_stimulus_freq *Hz
        
        
    return rates


#==============================================================================
#==============================================================================


#==============================================================================
#=============================verify_stimulation===============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function chooses a given subset of neurons and stimulate them in the
# beginning of the simulations
#------------------------------------------------------------------------------

def verify_stimulation(t):
    rates=np.zeros([n])*Hz    
    
    if t in stimul_times:        
        #input_index = floor(n*rand(round(n*qqi)))
        #input_index = input_index.astype(int)        
        input_index = [neuron_indices[i] for i, x in enumerate(stimul_times*1000) if x*ms == t]
        rates[input_index]=ones(round(n*qqi))*input_stimulus_freq *Hz
        
        
    return rates
#==============================================================================
#==============================================================================

#==============================================================================
#=============================generate_activity================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function runs the neural networks and generatethe required neural
# activity. The Brian simulator is used for this part.
#------------------------------------------------------------------------------

def generate_neural_activity(NeuralNetwork,running_period,S_time_file_base,frac_input_neurons,cascade_count,generate_data_mode):
    
    #--------------------------Initializing Variables--------------------------
    global n
    global qqi
    global input_stimulus_freq
    global stimul_times
    global neuron_indices
        
    qqi = frac_input_neurons
    input_stimulus_freq = 20000               # The frequency of spikes by the neurons in the input layer (in Hz)
    import brian
    n_exc = int(NeuralNetwork.n_exc_array[0])
    n_inh = int(NeuralNetwork.n_inh_array[0])
    n = n_exc + n_inh
    stimul_times = []
    neuron_indices = []
        
    if generate_data_mode == 'R':
        
        
        for j in range(0,n):
            #no_stim_points = int(running_period/3.2)
            no_stim_points = int(frac_input_neurons* running_period*(0.5+rand(1)/2))
            #no_stim_points = no_stim_points[0]
            times_neur = range(1,int(running_period))
            random.shuffle(times_neur)
            times_neur = times_neur[0:no_stim_points]
            stimul_times.extend(times_neur)
            temp_ind = j*np.ones([no_stim_points])
            neuron_indices.extend(temp_ind)
                    
        #stimul_times.sort()        
        for i in range(0,len(stimul_times)):
            stimul_times[i] = stimul_times[i] *ms
    else:
        no_stimulated_neurons = int(frac_input_neurons * n)
        temp = range(0,int(n))
        random.shuffle(temp)
        neuron_indices.extend(temp[0:no_stimulated_neurons])
        stimul_times.extend(0.1 *np.ones([no_stimulated_neurons]))
        for i in range(0,no_stimulated_neurons):
            stimul_times[i] = stimul_times[i] *ms        
        #neuron_indices = 
        
    #..............Retrieve the Parameters for the Neural Models...............
    eqs = NeuralNetwork.neural_model_eq[0]
    tau = NeuralNetwork.neural_model_eq[1]
    tau_e = NeuralNetwork.neural_model_eq[2]
    #..........................................................................
    
    #--------------------------------------------------------------------------
        
    #--------------------------Initialize the Network--------------------------
    neurons_list = {}
    delayed_connections = {}
    for l in range(0,NeuralNetwork.no_layers):
        temp_list = []            
        n_exc = int(NeuralNetwork.n_exc_array[l])
        n_inh = int(NeuralNetwork.n_inh_array[l])
        n = n_exc + n_inh
            
        neurons = NeuronGroup(n,model=eqs,threshold=5*mV,reset=0*mV,refractory=1*ms)
            
        neurons_list[str(l)] = list([neurons,n_exc,n_inh])    
    #--------------------------------------------------------------------------
        
    #-----------------------Connect the Layers Together------------------------
    for l_in in range(0,NeuralNetwork.no_layers):
            
        #....................Retrieve the Layers Parameters....................
        temp_list = neurons_list[str(l_in)]
        n_exc = temp_list[1]
        n_inh = temp_list[2]
        input_layer = temp_list[0]
        #......................................................................
            
        for l_out in range(l_in,NeuralNetwork.no_layers):
                
            #~~~~~~~~~~~~~~~~~~Retrieve the Network Parameters~~~~~~~~~~~~~~~~~
            temp_list = neurons_list[str(l_out)]
            output_layer = temp_list[0]
                
            ind = str(l_in) + str(l_out)
            main_layer = NeuralNetwork.Neural_Connections[ind]
            W = main_layer[0]
            D = main_layer[1]
            delay_max = main_layer[2]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
            #~~~~~~~~~~~~~~~~~~Fix the Connections and Delays~~~~~~~~~~~~~~~~~~
            C = DelayConnection(input_layer, output_layer,max_delay = delay_max*ms,delay = lambda i, j:delay_max * abs(sign(j-i))* rand(1) * ms)                
            C.connect(input_layer,output_layer,sparse.csc_matrix(W))                    
            C.set_delays(input_layer,output_layer,sparse.csc_matrix(D))

            delayed_connections[ind] = C
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #......................................................................
        
    #--------------------------------------------------------------------------
        
    #--------Create and Initialize the Dummy Input Layer for Stimulation-------
    temp_list = neurons_list['0']
    main_network = temp_list[0]
    
    n = int(NeuralNetwork.n_exc_array[0]) + int(NeuralNetwork.n_inh_array[0])
    
    input_dummy_layer=PoissonGroup(n,rates=initial_stimulation)    
    input_connections=Connection(input_dummy_layer,main_network,weight=lambda i,j:(1-abs(sign(i-j))),delay = 0*ms)
    #--------------------------------------------------------------------------
    
    #-------------Create the Network and Add the Necessary Monitors------------
    C = delayed_connections['00']
    
    
    net = Network(main_network,[C])
    net.add(input_connections)
    net.add(input_dummy_layer)
    
    Spike_monitors_list = {}
    Spike_monitors_list['dummy'] = SpikeMonitor(input_dummy_layer)
    Spike_monitors_list['l_0'] = SpikeMonitor(main_network)
    net.add(Spike_monitors_list['dummy'])
    net.add(Spike_monitors_list['l_0'])
    
    for l_in in range(1,NeuralNetwork.no_layers):
        temp_list = neurons_list[str(l_in)]
        neural_layer = temp_list[0]            
        net.add(neural_layer)    
            
        ind = 'l_' + str(l_in)
        Spike_monitors_list[ind] = SpikeMonitor(neural_layer)            
        net.add(Spike_monitors_list[ind])
            
    for l_in in range(0,NeuralNetwork.no_layers):
        for l_out in range(l_in,NeuralNetwork.no_layers):
            ind = str(l_in) + str(l_out)
            C = delayed_connections[ind]
                
            net.add(C)
                
                
    net.run(running_period * ms)        
    
    #pdb.set_trace()
    
    print Spike_monitors_list['dummy'].nspikes, "spikes in dummy layer"        
    for l_in in range(0,NeuralNetwork.no_layers):
        ind = 'l_' + str(l_in)
        print Spike_monitors_list[ind].nspikes, "spikes in layer %s" %str(l_in)            
    #--------------------------------------------------------------------------
    
    #----------------------Save Spike Times to the File------------------------    
    for l_in in range(0,NeuralNetwork.no_layers):
        file_name = S_time_file_base + '_l_' + str(l_in) +'.txt'
        S_time_file = open(file_name,'a+')
            
        ind = 'l_' + str(l_in) 
        SS = Spike_monitors_list[ind].spikes          
            
            
        for l in range(0,len(SS)):
            item = SS[l]
                
            if (len(item)>1):
                a = item[0]
                b = item[1]
                b = b.astype(float)
                S_time_file.write("%d \t %f \n" %(a,b))
    
        S_time_file.write("-2 \t -2 \n")
        
        S_time_file.close()
    #--------------------------------------------------------------------------    
    
    #---------------Reinitialize the Clocks for Spike Timings------------------
    Spike_monitors_list['dummy'].source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0    
    for l_in in range(0,NeuralNetwork.no_layers):
        ind = 'l_' + str(l_in)
        Spike_monitors_list[ind].source.clock.reinit()            
    #--------------------------------------------------------------------------
    
    #-------------Save the Connectivity and Delays in the Network--------------
    Neural_Connections_Out = {}
    for l_in in range(0,NeuralNetwork.no_layers):
        for l_out in range(l_in,NeuralNetwork.no_layers):
            ind = str(l_in) + str(l_out)
            C = delayed_connections[ind]
            WW = C.W.todense()
            DD = C.delay.todense()
            

            Neural_Connections_Out[ind] = list([WW,DD])
    #--------------------------------------------------------------------------
        
    #--------------------Reset Everything to Rest Conditions-------------------
    clear(spikequeue)
    #--------------------------------------------------------------------------
    
    return Neural_Connections_Out
#==============================================================================
#==============================================================================


#==============================================================================
#=============================beliefs_to_binary================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function transforms the beliefs matrix to a binary matrix, with +1,-1,
# and 0 entries.
#------------------------------------------------------------------------------

def beliefs_to_binary(binary_mode,W_inferred,params,compensate_flag):
    
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
    if (lll>1):
        W_binary = np.zeros([n,m])
        if ( (binary_mode == 4) or (binary_mode == 5)):
            centroids = np.zeros([n,3])
        W_binary.fill(0)
        
    else:
        W_binary = np.zeros([n,1])
        W_binary.fill(0)
        if ( (binary_mode == 4) or (binary_mode == 5)):
            centroids = np.zeros([1,3])
    #--------------------------------------------------------------------------
    
    #----------------Binary Mode 1: Probabilistic Thresholding-----------------
    if (binary_mode == 1):
        
        #..................Get the Binarification Parameters...................
        q = params[5]
        r = params[6]        
        params_bin = np.hstack([params,n])        
        #......................................................................
        
        #..,...................Determine the Thresholds........................        
        thrs = determine_binary_threshold('p',params_bin,[])
        thr_exc = thrs[0]
        thr_inh = thrs[2]
        #......................................................................
        
        #...........Adjust the Thresholds According to Real Time Data..........
        #thr_exc_adjusted = thr_exc * q/frac_input_neurons
        #thr_inh_adjusted = thr_inh * q/frac_input_neurons
            
        thr_exc_adjusted = thr_exc  * (2*r-1)/(2*p_W_zero_1-1)
        thr_inh_adjusted = thr_inh * (2*r-1)/(2*p_W_zero_1-1)
        #......................................................................
        
        #...................Transform the Graph into Binary....................        
        temp = (W_inferred > thr_exc_adjusted)
        temp = temp.astype(int)            
        W_binary = W_binary + temp
            
        temp = (W_inferred < thr_inh_adjusted)
        temp = temp.astype(int)
        W_binary = W_binary - temp
        #......................................................................
    #--------------------------------------------------------------------------
    
    #----------------Binary Mode 2: Sorting-Based Thresholding-----------------
    elif (binary_mode == 2):
        
        #..................Get the Binarification Parameters...................
        q = params[0]
        r = params[1]
        p_exc = params[2]
        p_inh = params[3]
        Delta = params[4]
        theta = params[5]
        #......................................................................
        
        #..........................Recurrent Networks..........................
        if (lll>1):

            for i in range(0,n):
                #~~Go Over Each Neuron and Pick the Highest and Lowest Beliefs~
                w_temp = W_inferred[:,i]
                w_temp_s = np.sort(W_inferred[:,i])
                ind = np.argsort(w_temp)
                w_temp = np.zeros(n)
                w_temp[ind[0:int(round(p_inh*n))+1]] = -1            
                w_temp[ind[len(ind)-int(round(p_exc*n))+1:len(ind)+1]] = 1            
                W_binary[:,i] = w_temp
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
        #......................................................................
        
        #.........................FeedForward Networks.........................
        else:
            ind = np.argsort(W_inferred)
            w_temp = np.zeros([n])
            w_temp[ind[0:int(round(p_inh*n))+1]] = -1
            w_temp[ind[len(ind)-int(round(p_exc*n))+1:len(ind)+1]] = 1            
            W_binary = w_temp
        #......................................................................
    
    #--------------------------------------------------------------------------
    
    #--------------Binary Mode 3: Neuron-Type-Based Thresholding---------------
    elif (binary_mode == 3):
        
        p_exc = params[2]
        p_inh = params[3]
        connection_prob = p_exc + p_inh
        #........Sum the Beleifs for the Outgoing Links of Each Neuron.........
        total_belief = np.sum(W_inferred,axis=1)
        temp1 = np.sort(total_belief)
        ind = np.argsort(total_belief)
        #......................................................................
        
        #....Pick the Excitatory Neurons and Set Their Outgoing Links to +1....
        for i in ind[len(ind)-int(round(p_exc*n))+1:len(ind)+1]:
            w_temp = W_inferred[i,:]
            w_temp_s = np.sort(W_inferred[i,:])
            ind2 = np.argsort(w_temp)
            w_temp = np.zeros(n) 
            w_temp[ind2[len(ind2)-int(round(connection_prob*n))+1:len(ind2)+1]] = 1
            W_binary[i,:] = w_temp
        #......................................................................
        
        #....Pick the Inhibitory Neurons and Set Their Outgoing Links to +1....
        for i in ind[0:int(round(p_inh*n))+1]:
            w_temp = W_inferred[i,:]
            w_temp_s = np.sort(W_inferred[i,:])
            ind2 = np.argsort(w_temp)
            w_temp = np.zeros(n) 
            w_temp[ind2[0:int(round(connection_prob*n))+1]] = -1
            W_binary[i,:] = w_temp
        #......................................................................
    
    #--------------------------------------------------------------------------
    
    #--------------Binary Mode 4: Clustering-Based Thresholding----------------
    elif (binary_mode == 4):

        #......................Determine the Thresholds........................
        fixed_inds = params[2]
        for i in range(0,m):            
            W_inferred_temp = copy.deepcopy(W_inferred[:,i])            
            temp = np.ma.masked_array(W_inferred_temp,mask= fixed_inds[:,i])
            masked_inds = np.nonzero((temp.mask).astype(int))
            masked_vals = temp.compressed()
                
            params = params[0:2]
            if sum(sum(abs(masked_vals))):                
                thr_inh,thr_zero,thr_exc = determine_binary_threshold('c',params,masked_vals)
                centroids[i,:] = [thr_inh,thr_zero,thr_exc]
            #print centroids[i,:]
        #......................................................................
        
        #...................Transform the Graph to Binary......................
                W_temp,res = vq(masked_vals,np.array([thr_inh,thr_zero,thr_exc]))
                W_temp = W_temp - 1
            else:                
                W_temp = np.zeros(masked_vals.shape)

        #----------------Role Back Values to Unmasked Indices--------------
            mask_counter = 0
            for j in range(0,n):                
                if j in masked_inds[0]:                    
                    W_binary[j,i] = sign(W_inferred[j,i])
                else:
                    W_binary[j,i] = W_temp[mask_counter]
                    mask_counter = mask_counter + 1
            #------------------------------------------------------------------    

        #......................................................................
    #--------------------------------------------------------------------------
    
    #----Binary Mode 5: Clustering-Based Thresholding Excluding Diagnoals------
    elif (binary_mode == 5):

        #......................Determine the Thresholds........................
        if (lll>1):        
            for i in range(0,n):
                ww = W_inferred[0:i,i]
                ww = np.append(ww,W_inferred[i+1:len(W_inferred[i,:]),i])
                thr_inh,thr_zero,thr_exc = determine_binary_threshold('c',params,ww)
                #pdb.set_trace()
                centroids[i,:] = [thr_inh,thr_zero,thr_exc]
                #print centroids[i,:]
        #......................................................................
        
        #...................Transform the Graph to Binary......................
                W_temp,res = vq(W_inferred[:,i],np.array([thr_inh,thr_zero,thr_exc]))
                W_temp = W_temp - 1
                W_binary[:,i] = W_temp
                #pdb.set_trace()
        else:
            ww = W_inferred.ravel()
            thr_inh,thr_zero,thr_exc = determine_binary_threshold('c',params,ww)
            centroids = [thr_inh,thr_zero,thr_exc]
            W_temp,res = vq(ww,np.array([thr_inh,thr_zero,thr_exc]))
            W_binary = W_temp - 1
        #......................................................................
    #--------------------------------------------------------------------------
    
    #--------Binary Mode 6: Whole Matrix Clustering-Based Thresholding---------
    elif (binary_mode == 6):
        ww = W_inferred.ravel()
        thr_inh,thr_zero,thr_exc = determine_binary_threshold('c',params,ww)
        centroids = [thr_inh,thr_zero,thr_exc]

        W_temp,res = vq(ww,np.array([thr_inh,thr_zero,thr_exc]))
        W_temp = W_temp - 1
        W_binary = W_temp.reshape([n,m])
        
    #--------------------------------------------------------------------------
    
    #-----Binary Mode 7: Only Assigning Those Edges That We Are Sure About-----
    elif (binary_mode == 7):
        
        
        
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
    if compensate_flag:
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
#=============================calucate_accuracy================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function compares the inferred binary matrix and compare it with the
# original graph to calculate the accruacy (recall and precision) of the
# algorithm
#------------------------------------------------------------------------------

def calucate_accuracy(W_binary,W):
    
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
#===========================verify_neural_activity=============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function runs the neural networks and re-generate neural activity for
# a given set of stimulated neurons and a network with a given connectivity
# matrix. The Brian simulator is used for this part.
#------------------------------------------------------------------------------

def verify_neural_activity_simplified(W,in_spikes,theta):
    
    #--------------------------Initializing Variables--------------------------    
    out_sps = (0.5*(1+np.sign(np.dot(W.T,(in_spikes>0).astype(int))-theta+0.00002)))
    return out_sps
    
    



def verify_neural_activity(NeuralNetwork,Network_in,running_period,frac_input_neurons,stimulated_neurons,stimulation_times):
    #--------------------------Initializing Variables--------------------------
    
    global stimul_times
    global neuron_indices
    global qqi
    global n
    global input_stimulus_freq
    import brian
    
    qqi = frac_input_neurons
    stimul_times = stimulation_times
    neuron_indices = stimulated_neurons
    input_stimulus_freq = 20000               # The frequency of spikes by the neurons in the input layer (in Hz)    
    
    
    
    for i in range(0,len(stimul_times)):
        stimul_times[i] = stimul_times[i] *ms
        
    #..............Retrieve the Parameters for the Neural Models...............
    eqs = NeuralNetwork.neural_model_eq[0]
    tau = NeuralNetwork.neural_model_eq[1]
    tau_e = NeuralNetwork.neural_model_eq[2]
    #..........................................................................
    
    #--------------------------------------------------------------------------
        
    
        
    #-----------------------Connect the Layers Together------------------------
    n_in = Network_in['n_in']
    n = n_in
    n_out = Network_in['n_out']
    W = Network_in['W']
    D = Network_in['D']
    delay_max = Network_in['d_max']
    
    input_layer = NeuronGroup(n_in,model=eqs,threshold=5*mV,reset=0*mV,refractory=1*ms)
    output_layer = NeuronGroup(n_out,model=eqs,threshold=5*mV,reset=0*mV,refractory=1*ms)
        
    C = DelayConnection(input_layer, output_layer,max_delay = delay_max*ms,delay = lambda i, j:delay_max * abs(sign(j-i))* rand(1) * ms)
    
    C.connect(input_layer,output_layer,sparse.csc_matrix(W))
    C.set_delays(input_layer,output_layer,sparse.csc_matrix(D))
    #..........................................................................
        
    #--------------------------------------------------------------------------
        
    #--------Create and Initialize the Dummy Input Layer for Stimulation-------    
    input_dummy_layer=PoissonGroup(n_in,rates=verify_stimulation)    
    input_connections=Connection(input_dummy_layer,input_layer,weight=lambda i,j:(1-abs(sign(i-j))),delay = 0*ms)
    #--------------------------------------------------------------------------
    
    #-------------Create the Network and Add the Necessary Monitors------------    
    net = Network(input_layer,[C])
    net.add(input_connections)
    net.add(input_dummy_layer)
    net.add(output_layer)
    
    Spike_monitors_list = {}
    Spike_monitors_list['dummy'] = SpikeMonitor(input_dummy_layer)
    Spike_monitors_list['l_0'] = SpikeMonitor(input_layer)
    Spike_monitors_list['l_1'] = SpikeMonitor(output_layer)            
    net.add(Spike_monitors_list['dummy'])
    net.add(Spike_monitors_list['l_0'])    
    net.add(Spike_monitors_list['l_1'])

    net.run(running_period * ms)        
        
    
    #print Spike_monitors_list['dummy'].nspikes, "spikes in dummy layer"            
    #print Spike_monitors_list['l_0'].nspikes, "spikes in pre-synaptic neurons"
    #print Spike_monitors_list['l_1'].nspikes, "spikes in output synaptic neuron"
    #--------------------------------------------------------------------------
    
    #----------------------Save Spike Times to the File------------------------    
    out_spike = 0
    SS = Spike_monitors_list['l_1'].spikes
    out_spike = np.zeros([1,n_out])
    if len(SS):
        for item in SS:            
            out_spike[0,item[0]] = item[1]        
    #--------------------------------------------------------------------------
    
    #---------------Reinitialize the Clocks for Spike Timings------------------
    Spike_monitors_list['dummy'].source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0    
    Spike_monitors_list['l_0'].source.clock.reinit()
    Spike_monitors_list['l_1'].source.clock.reinit()
    #--------------------------------------------------------------------------
    
    #-------------Save the Connectivity and Delays in the Network--------------    
    WW = C.W.todense()
    DD = C.delay.todense()
    Neural_Connections_Out = list([WW,DD])
    #--------------------------------------------------------------------------
    
    return Neural_Connections_Out,out_spike


    
#==============================================================================
#==============================================================================


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
#========================THE BASIC INFERENCE ALGORITHM=========================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function truncates performs different inference methods.
#------------------------------------------------------------------------------
def inference_alg_per_layer(in_spikes,out_spikes,inference_method,inferece_params,W_estim,location_flag):
           
    #------------------------------Initialization------------------------------
    s = in_spikes.shape
    n = s[0]
    TT = s[1]
    
    s = out_spikes.shape
    m = s[0]
    
    if (TT != s[1]):
        print('Error in the number of samples!')
        sys.exit()
        
    W_inferred = np.zeros([n,m])
    Updated_Vals = np.zeros([n,m])
    W_inferred.fill(0)
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
                    W_inferred_orig[i,j] = 0
                    
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
        #......................................................................        
        
        #..............................Initializations.........................        
        range_tau = range(0,max_itr_opt)
        
        cost = np.zeros([len(range_tau)])        
        #......................................................................
        
        #................Iteratively Update the Connectivity Matrix............
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
            for cascade_count in range(0,TT):                
                x = in_spikes[:,cascade_count]
                x = x.reshape(n,1)
                if (location_flag == 1):
                    v = (x>0).astype(int)
                y = out_spikes[:,cascade_count]
                y = y.reshape(m,1)
                yy_predict = np.zeros([m,1])
                
                random.shuffle(neurons_ind)
                
                #~~~~~~~~~~~Upate the Incoming Weights to Each Neuron~~~~~~~~~~                
                for ijk2 in range(0,m):
                    ijk = neurons_ind[ijk2]
                    yy = y[ijk]                    
                    WW = W_inferred[:,ijk]
                    if (location_flag == 0):
                        if inference_method == 2:
                            v = (x>0).astype(int)
                        else:
                            if yy > 0:
                                v = (x<yy).astype(int)
                            else:
                                v = (x>0).astype(int)
                            v = np.multiply(v,(x>0).astype(int))
                    
                    y_predict = 0.5*(1+np.sign(np.dot(WW,v)-theta+0.00002))
                    upd_val = np.dot(y_predict - np.sign(yy),v.T)
                    W_inferred[:,ijk] = W_inferred[:,ijk] - alpha*np.multiply(upd_val,1-fixed_ind[:,ijk])
                    Updated_Vals[:,ijk] = Updated_Vals[:,ijk] + np.sign(abs(v.T))
                    
                    cost[ttau] = cost[ttau] + sum(pow(y_predict - (yy>0.0001).astype(int),2))
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~~~~~~~~~Saturate the Weights~~~~~~~~~~~~~~~~~~~~                
                ter = W_inferred < 0.001
                W_inferred = np.multiply(ter.astype(int),W_inferred) + 0.001*(1-ter.astype(int))
                ter = W_inferred >- 0.001
                W_inferred = np.multiply(ter.astype(int),W_inferred) - 0.001*(1-ter.astype(int))
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
    #--------------------------------------------------------------------------
    
    #---------------------The Correlation-based Algorithm----------------------
    elif (inference_method == 0):
        Delta = inferece_params[0]#/float(inferece_params[1])
        theta = inferece_params[1]
        if location_flag == 1:            
            out_spikes_estimated = np.sign((np.dot(W_inferred.T,in_spikes)>theta).astype(int))
            out_spikes_estimated = out_spikes_estimated.astype(int)
            W_temp = np.dot(in_spikes,(out_spikes-out_spikes_estimated).T)*Delta
        else:
            m,T = out_spikes.shape
            n,T = in_spikes.shape
            W_temp = np.zeros([n,m])
            for j in range(0,m):
                y = out_spikes[j,:]
                v = np.zeros([n,T])
                for i in range(0,n):                    
                    v[i,:] = (in_spikes[i,:]<y).astype(int)                
                    v[i,:] = v[i,:] + np.multiply((in_spikes[i,:]>0).astype(int),y==0)
                    v[i,:] = np.multiply(v[i,:],(in_spikes[i,:]>0).astype(int))
                #pdb.set_trace()
                Updated_Vals[:,j] = Updated_Vals[:,j] + sum(abs(v),axis=1)    
                out_spikes_estimated = np.sign((np.dot(W_inferred[:,j].T,v)>theta).astype(int))
                out_spikes_estimated = out_spikes_estimated.astype(int)                
                W_temp[:,j] = np.dot(v,(np.sign(y)-out_spikes_estimated).T)*Delta#,(0.0001+sum(v,axis=1)*sum(y)/float(T)))
                
        #pdb.set_trace()
        W_inferred = np.multiply(1-fixed_ind,W_temp) + np.multiply(fixed_ind,W_inferred)
    #--------------------------------------------------------------------------
    
    #--------------------------The CCF-based Algorithm----------------------
    elif (inference_method == 4):
        d_window = inferece_params[0]           
        d_range = range(1,d_window+1)        
        D_estimated = np.zeros([n,m])
        W_temp = np.zeros([n,m])
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
    
    #--------------The Event-Based Algorithm with Integration------------------
    elif (inference_method == 5):
        alpha0 = inferece_params[0]
        sparse_thr_0 = inferece_params[1]
        sparsity_flag = inferece_params[2]
        theta = inferece_params[3]
        
        if (norm(inferece_params[4])):
            W_inferred = inferece_params[4]            
            fixed_ind = (W_inferred > 0)
            fixed_ind = fixed_ind.astype(int)
        else:
            fixed_ind = np.zeros([n,m])
        
        sim_window = inferece_params[5]
        fixed_ind = np.zeros([n,m])
        W_inferred_orig = W_inferred                
        range_tau = range(0,1000)
        
        cost = sum(in_spikes)*np.ones([len(range_tau)])
        TT = int(TT/float(sim_window))

        #-----------------Construct the Cumulative Sum-----------------
        
        cumul_sum_20 = np.zeros([m,sim_window])
        cumul_sum_10 = np.zeros([m,sim_window])
        binned_spikes_tot = np.zeros([m,sim_window])

        
        
        for t in range(1,int(sim_window)-1):
            cumul_sum_20[:,t] = sum(in_spikes[:,max(0,t-20):t],axis=1)
            cumul_sum_10[:,t] = sum(in_spikes[:,max(0,t-10):t],axis =1)
            y_binned = out_spikes[:,t:min(t+10,int(sim_window))]
            y_binned = sum(y_binned,axis = 1)
            binned_spikes_tot[:,t] = (y_binned>0).astype(int)
                    

        #--------------------------------------------------------------

        print 'Integration finished'
        for ttau in range_tau:
            
            #-------------------Adjust Optimization Parameters----------
            temp = 0
            alpha = alpha0/float(1+log(ttau+1))
            sparse_thr = sparse_thr_0/float(1+log(ttau+1))
            column_ind = range(0,int(m))
            random.shuffle(column_ind)
            
            #-----------------------------------------------------------
            mem_pot = np.zeros([sim_window])
            binned_spikes =  np.zeros([sim_window])
            for ijk2 in range(0,m):
                ijk = column_ind[ijk2]
                
                #temp_cumul_sum = cumul_input_sum[ijk,:]
                #temp_binned_y = binned_spikes[ijk,:]
                

                WW = W_inferred[:,ijk]
                
                firing_ind = np.nonzero(out_spikes[ijk,:])
                firing_ind = firing_ind[0]
                firing_ind = np.hstack([0,firing_ind])
                firing_ind = np.hstack([firing_ind,int(sim_window)])
                mem_pot.fill(0)
                binned_spikes.fill(0)
                update_flag = 0
                for l in range(0,len(firing_ind)-1):
                    for t in range(firing_ind[l]+1,firing_ind[l]+11):
                        #pdb.set_trace()
                        mem_pot[t] = np.dot(WW,cumul_sum_10[:,t])
                        binned_spikes[t] = binned_spikes_tot[ijk,t] 

                    for t in range(firing_ind[l]+11,firing_ind[l+1]):
                        mem_pot[t] = np.dot(WW,cumul_sum_20[:,t])
                        binned_spikes[t] =  binned_spikes_tot[ijk,t]

                    temp_binned_y = binned_spikes[firing_ind[l]+1:t+1]
                    y_predict = 0.5*(1+np.sign(mem_pot[firing_ind[l]+1:t+1]-theta+0.00002))
                    fire_ind = np.nonzero(y_predict)
                    fire_ind = fire_ind[0]
                    #if ttau>0:
                    #    pdb.set_trace()
                    #if n
                    if norm(fire_ind):
                        iind = fire_ind[0]
                        v = cumul_sum_20[:,iind]

                        
                        
                        if (y_predict[iind] != temp_binned_y[iind]):
                            cost[ttau] = cost[ttau] + 1
                            
                            #upd_val = np.dot(y_predict[iind] - temp_binned_y[iind],v.T)
                            #W_inferred[:,ijk] = W_inferred[:,ijk] - alpha*upd_val
                            firing_ind_v = np.nonzero(v)
                            firing_ind_v = firing_ind_v[0]
                            
                            if norm(firing_ind_v):
                                random.shuffle(firing_ind_v)
                                ijl = firing_ind_v[0]
                            
                                upd_val = np.dot(y_predict[iind] - temp_binned_y[iind],v[ijl])
                                W_inferred[ijl,ijk] = W_inferred[ijl,ijk] - alpha*upd_val

                            
                            #print ijk
                                update_flag = 1
                                break
                            
                    else:
                        fire_ind = np.nonzero(temp_binned_y)
                        fire_ind = fire_ind[0]
                        
                        if norm(fire_ind):
                            iind = fire_ind[0]
                            v = cumul_sum_20[:,iind]
                            
                            
                            if (y_predict[iind] != temp_binned_y[iind]):
                                firing_ind_v = np.nonzero(v)
                                firing_ind_v = firing_ind_v[0]
                                
                                if norm(firing_ind_v):
                                    random.shuffle(firing_ind_v)
                                    ijl = firing_ind_v[0]
                                    upd_val = np.dot(y_predict[iind] - temp_binned_y[iind],v[ijl])
                                    W_inferred[ijl,ijk] = W_inferred[ijl,ijk] - alpha*upd_val
                                #upd_val = np.dot(y_predict[iind] - temp_binned_y[iind],v.T)
                                #W_inferred[:,ijk] = W_inferred[:,ijk] - alpha*upd_val
                                #print ijk
                                    update_flag = 1
                                    break
                            else:
                                cost[ttau] = cost[ttau] - 1

                #------------Saturate the Weights to +0.001 and -0.001------------                        
                W_inferred[:,ijk] = np.multiply(1-fixed_ind[:,ijk],W_inferred[:,ijk]) + np.multiply(fixed_ind[:,ijk],W_inferred_orig[:,ijk]) 
                ter = W_inferred[:,ijk] < 0.001
                W_inferred[:,ijk] = np.multiply(ter.astype(int),W_inferred[:,ijk]) + 0.001*(1-ter.astype(int))
                ter = W_inferred[:,ijk] >- 0.001
                W_inferred[:,ijk] = np.multiply(ter.astype(int),W_inferred[:,ijk]) - 0.001*(1-ter.astype(int))
                #-----------------------------------------------------------------

            print cost[ttau]
            print sum(abs(W_inferred))
    
    #-----------------------The Hebbian-based Algorithm------------------------
    elif (inference_method == 1):
        Delta = inferece_params[0]/float(T)
        W_temp = np.dot(in_spikes,out_spikes.T)*Delta
        W_inferred = np.multiply(1-fixed_ind,W_temp) + np.multiply(fixed_ind,W_inferred)                
    #--------------------------------------------------------------------------
    
    else:
        print('Error! Invalid inference method.')
        sys.exit()
    
    
    return W_inferred,cost,Updated_Vals
    
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
                    tt = round(10000*S_times[l,1])-1                
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
            
            for l in range(0,s[0]):
                neuron_count = int(S_times[l,0])        
                if (neuron_count >= 0):                    
                    tt = round(10000*S_times[l,1])-1                
                    in_spikes[neuron_count,tt] = 1
                    tot_spikes[neuron_count+n_so_far,tt] = 1
                    
            n_so_far = n_so_far + n
            Neural_Spikes[str(l_in)] = list([[],[],in_spikes])
            Neural_Spikes['tot'] = tot_spikes
            print(sum(in_spikes))
    
    return Neural_Spikes
#==============================================================================
#==============================================================================


#==============================================================================
#=========================BELIEF QUALITY ASSESSMENT============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function calculates the quality of the computed beliefs about the neural
# graph
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

    
    return np.hstack([B_exc_mean,B_void_mean,B_inh_mean]),np.hstack([B_exc_min,B_void_max,B_void_min,B_inh_max])
#==============================================================================
#==============================================================================


#==============================================================================
#==========================REAL TO DISCRETE SPIKES=============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function transforms a spike train that has real-valued recorded spike
# times to one that has discrete values. Example: 0.0002 --> [001000100]
#------------------------------------------------------------------------------
def real_to_discrete_spikes(in_spikes,t_max):
    ss = in_spikes.shape
    n = ss[0]
    T = ss[1]
    output_spikes = np.zeros([n,t_max*T])
    
    fire_ind = np.nonzero(in_spikes)
    l = len(fire_ind[0])
    for i in range(0,l):
        val = in_spikes[fire_ind[0][i],fire_ind[1][i]]
        val = round(10000*val)
        output_spikes[fire_ind[0][i],fire_ind[1][i]*t_max + val] = 1
        
    return output_spikes    
#==============================================================================
#==============================================================================
