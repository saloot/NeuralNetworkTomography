#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
from brian import *
from scipy import sparse
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
        n = params[6]
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
        centroids,res = kmeans(obs,3,iter=30)
        ss = np.sort(centroids)
        val_inh = ss[0]
        val_exc = ss[2]
        thr_zero = ss[1]
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
    if t < 0.00001 * ms:
        input_index = floor(n*rand(round(n*qqi)))
        input_index = input_index.astype(int)
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

def generate_neural_activity(network_type,Neural_Connections,running_period,S_time_file_base,neural_model_eq,frac_input_neurons,cascade_count,no_layers):
    
    #--------------------------Initializing Variables--------------------------
    global n
    global qqi
    global input_stimulus_freq
    
    qqi = frac_input_neurons
    input_stimulus_freq = 20000               # The frequency of spikes by the neurons in the input layer (in Hz)
    import brian
    
    
    
    #..............Retrieve the Parameters for the Neural Models...............
    eqs = neural_model_eq[0]
    tau = neural_model_eq[1]
    tau_e = neural_model_eq[2]
    #..........................................................................
    
    #--------------------------------------------------------------------------
    
    #---------------------------The Recurrent Network--------------------------
    if (network_type == 'R'):
        
        #....................Retrieve the Network Parameters...................
        neural_layers = []
        n_exc_array = Neural_Connections['n_exc']
        n_inh_array = Neural_Connections['n_inh']
        
        if (no_layers == 1):
            n_exc = n_exc_array[0]
            n_inh = n_inh_array[0]
        n = n_exc + n_inh
        main_layer = Neural_Connections['00']
        We = main_layer[0]
        De = main_layer[1]
        Wi = main_layer[2]
        Di = main_layer[3]
        delay_max = main_layer[4]
        #......................................................................
        
        #........................Initialize the Network......................
        main_network=NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)    # The whole neural network
        Pe = main_network.subgroup(n_exc)                                   # The excitatory sub-group of neurons
        Pi = main_network.subgroup(n_inh)                                   # The inhibitory sub-group of neurons
        
        Ce = DelayConnection(Pe, main_network,max_delay = delay_max*ms,delay = lambda i, j:delay_max * rand(1) * ms)
        Ci = DelayConnection(Pi, main_network,max_delay = delay_max*ms,delay = lambda i, j:delay_max * rand(1) * ms)
        neural_layers.append(main_network)
        #......................................................................
        
        #....................Fix the Connections and Delays...................
        Ce.connect(Pe,main_network,sparse.csc_matrix(We))    
        Ci.connect(Pi,main_network,sparse.csc_matrix(Wi))
    
        Ce.set_delays(Pe,main_network,sparse.csc_matrix(De))
        Ci.set_delays(Pi,main_network,sparse.csc_matrix(Di))
        #......................................................................
        
    #--------------------------------------------------------------------------
    
    #--------------------------The FeedForward Network-------------------------
    elif (network_type == 'F'):
        
        #....................Retrieve the Network Parameters...................
        neurons_list = {}
        delayed_connections_exc = {}
        delayed_connections_inh = {}        
        n_exc_array = Neural_Connections['n_exc']
        n_inh_array = Neural_Connections['n_inh']
        #......................................................................
        
        #........................Initialize the Network........................
        for l in range(0,no_layers):
            temp_list = []
            ind = str
            n_exc = int(n_exc_array[l])
            n_inh = int(n_inh_array[l])
            n = n_exc + n_inh
            if (l == 0):
                neurons = NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)
            else:
                neurons = NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)
            
            Pe = neurons.subgroup(n_exc)
            Pi = neurons.subgroup(n_inh)
            
            
            neurons_list[str(l)] = list([neurons,Pe,Pi,n_exc,n_inh])
            
            
            
        output_neuron = NeuronGroup(1,model=eqs,threshold=10*mV,reset=0*mV)        
        neurons_list[str(l+1)] = list([output_neuron])
        #......................................................................
        
        #.....................Connect the Layers Together......................
        for l_in in range(0,no_layers):
            
            #~~~~~~~~~~~~~~~~~~Retrieve the Layers Parameters~~~~~~~~~~~~~~~~~~
            temp_list = neurons_list[str(l_in)]
            Pe = temp_list[1]
            Pi = temp_list[2]
            n_exc = temp_list[3]
            n_inh = temp_list[4]
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            for l_out in range(l_in,no_layers):
                
                #~~~~~~~~~~~~~~~~Retrieve the Network Parameters~~~~~~~~~~~~~~~
                temp_list = neurons_list[str(l_out+1)]
                output_layer = temp_list[0]
                
                ind = str(l_in) + str(l_out)
                main_layer = Neural_Connections[ind]
                We = main_layer[0]
                De = main_layer[1]
                Wi = main_layer[2]
                Di = main_layer[3]
                delay_max = main_layer[4]
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                #~~~~~~~~~~~~~~~~Fix the Connections and Delays~~~~~~~~~~~~~~~~
                Ce = DelayConnection(Pe, output_layer,max_delay = delay_max*ms,delay = lambda i, j:delay_max * rand(1) * ms)
                Ci = DelayConnection(Pi, output_layer,max_delay = delay_max*ms,delay = lambda i, j:delay_max * rand(1) * ms)
                
                Ce.connect(Pe,output_layer,sparse.csc_matrix(We))    
                Ci.connect(Pi,output_layer,sparse.csc_matrix(Wi))
    
                Ce.set_delays(Pe,output_layer,sparse.csc_matrix(De))
                Ci.set_delays(Pi,output_layer,sparse.csc_matrix(Di))
                
                
                delayed_connections_exc[ind] = Ce
                delayed_connections_inh[ind] = Ci
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #......................................................................
        
    #--------------------------------------------------------------------------
        
    #--------Create and Initialize the Dummy Input Layer for Stimulation-------
    temp_list = neurons_list['0']
    main_network = temp_list[0]
    
    n = int(n_exc_array[0]) + int(n_inh_array[0])
    
    #main_network.v = randint(1,3,n)-1
    #print sum(main_network.v)
    inputer_dummy_layer=PoissonGroup(n,rates=initial_stimulation)    
    input_connections=Connection(inputer_dummy_layer,main_network,weight=lambda i,j:(1-abs(sign(i-j))),delay = 0*ms)
    
    #p1= PoissonInput(main_network, N=n, rate=1, weight=1, state='I')
    #--------------------------------------------------------------------------
    
    #-------------Create the Network and Add the Necessary Monitors------------
    Ce = delayed_connections_exc['00']
    Ci = delayed_connections_inh['00']
    
    net = Network(main_network,[Ce,Ci])
    net.add(input_connections)
    net.add(inputer_dummy_layer)
    #net.add(p1)
    
    Spike_monotos_list = {}
    Spike_monotos_list['dummy'] = SpikeMonitor(inputer_dummy_layer)
    Spike_monotos_list['l_1'] = SpikeMonitor(main_network)
    net.add(Spike_monotos_list['dummy'])
    net.add(Spike_monotos_list['l_1'])
    if (network_type == 'F'):
        for l_in in range(1,no_layers+1):
            temp_list = neurons_list[str(l_in)]
            neural_layer = temp_list[0]            
            net.add(neural_layer)    
            
            ind = 'l_' + str(l_in+1)
            Spike_monotos_list[ind] = SpikeMonitor(neural_layer)            
            net.add(Spike_monotos_list[ind])
            
        for l_in in range(0,no_layers):
            for l_out in range(l_in,no_layers):
                ind = str(l_in) + str(l_out)
                Ce = delayed_connections_exc[ind]
                Ci = delayed_connections_inh[ind]
                net.add(Ce)
                net.add(Ci)
                
    net.run(running_period * ms)        
    
    if (network_type == 'R'):
        print Spike_monotos_list['dummy'].nspikes, "spikes in dummy layer"
        print Spike_monotos_list['l_1'].nspikes, "spikes in output layer"
    elif (network_type == 'F'):
        print Spike_monotos_list['dummy'].nspikes, "spikes in dummy layer"        
        for l_in in range(0,no_layers+1):
            ind = 'l_' + str(l_in+1)
            print Spike_monotos_list[ind].nspikes, "spikes in layer %s" %str(l_in+1)
            
    #--------------------------------------------------------------------------
    
    #----------------------Save Spike Times to the File------------------------
    if (network_type == 'R'):
        SS = M_l2.spikes
        for l in range(0,len(SS)):
            item = SS[l]
            a = item[0]
            b = item[1]
            b = b.astype(float)
            S_time_file.write("%d \t %f \n" %(a,b))
    
        S_time_file.write("-2 \t -2 \n")
    
    if (network_type == 'F'):
        for l_in in range(0,no_layers+1):
            file_name = S_time_file_base + '_l_' + str(l_in) +'.txt'
            S_time_file = open(file_name,'a')
            
            ind = 'l_' + str(l_in+1) 
            SS = Spike_monotos_list[ind].spiketimes          
            
            
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
    Spike_monotos_list['dummy'].source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0    
    if (network_type == 'F'):
        for l_in in range(0,no_layers+1):
            ind = 'l_' + str(l_in+1)
            Spike_monotos_list[ind].source.clock.reinit()            
    #--------------------------------------------------------------------------
    
    #-------------Save the Connectivity and Delays in the Network--------------
    Neural_Connections_Out = {}
    for l_in in range(0,no_layers):
        for l_out in range(l_in,no_layers):
            ind = str(l_in) + str(l_out)
            Ce = delayed_connections_exc[ind]
            Ci = delayed_connections_inh[ind]
            WWe = Ce.W.todense()
            WWi = Ci.W.todense()
            DDe = Ce.delay.todense()
            DDi = Ci.delay.todense()

            Neural_Connections_Out[ind] = list([WWe,DDe,WWi,DDi])
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

def beliefs_to_binary(binary_mode,W_inferred,n,p_exc,p_inh,thea,T,Delta,params,compensate_flag):
    
    #---------------------Import Necessary Libraries---------------------------
    import numpy as np
    from scipy.cluster.vq import kmeans,whiten,kmeans2,vq
    #--------------------------------------------------------------------------
    
    #--------------------------Initializing Variables--------------------------
    centroids = []
    connection_prob = p_exc + p_inh
    a = W_inferred.shape    
    n = a[0]
    lll = len(a)
    #--------------------------------------------------------------------------
    
    #---------Determine the Type of Network: FeedForward or Recurrent----------
    if (lll>1):
        W_binary = np.zeros([n,n])
        W_binary.fill(0)
    else:
        W_binary = np.zeros([n])
        W_binary.fill(0)
    #--------------------------------------------------------------------------
    
    #----------------Binary Mode 1: Probabilistic Thresholding-----------------
    if (binary_mode == 1):
        
        #..................Get the Binarification Parameters...................
        q = params[0]
        r = params[1]
        #......................................................................
        
        #..,...................Determine the Thresholds........................
        params_bin = [p_exc,p_inh,theta,T,Delta,q,n]
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

        #..,...................Determine the Thresholds........................
        ww = W_inferred.ravel()
        thr_inh,thr_zero,thr_exc = determine_binary_threshold('c',params,obs)
        centroids = [thr_inh,thr_zero,thr_exc]
        #......................................................................
        
        #...................Transform the Graph to Binary......................
        W_temp,res = vq(ww,np.array([thr_inh,thr_zero,thr_exc]))
        W_temp = W_temp - 1
        #......................................................................
        
        #....Reshape the Binary Vector to Matrix in Case of Recurrent Graphs...
        if (lll>1):
            for i in range(0,n):
                W_binary[i,:] = W_binary[i,:] + W_temp[(i)*n:(i+1)*n]
        else:
            W_binary = W_temp
        #......................................................................
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
    if (lll>1):
        A = np.zeros([n,n])
        acc_plus = float(sum(sum(np.multiply(W_binary>A,W>A))))/float(sum(sum(W>A)))
        acc_minus = float(sum(sum(np.multiply(W_binary<A,W<A))))/float(sum(sum(W<A)))
        acc_zero = float(sum(sum(np.multiply(W_binary==A,W==A))))/float(sum(sum(W==A)))
        
        prec_plus = float(sum(sum(np.multiply(W_binary>A,W>A))))/float(sum(sum(W_binary>A)))
        prec_minus = float(sum(sum(np.multiply(W_binary<A,W<A))))/float(sum(sum(W_binary<A)))
        prec_zero = float(sum(sum(np.multiply(W_binary==A,W==A))))/float(sum(sum(W_binary==A)))
    #--------------------------------------------------------------------------
    
    #---------------Compute Accuracy for FeedForward Networks------------------
    else:
        A = np.zeros([n])
        acc_plus = float(sum(np.multiply(W_binary>A,W>A)))/float(sum(W>A))
        acc_minus = float(sum(np.multiply(W_binary<A,W<A)))/float(sum(W<A))
        acc_zero = float(sum(np.multiply(W_binary==A,W==A)))/float(sum(W==A))

        if (sum(W_binary>A)):
            prec_plus = float(sum(np.multiply(W_binary>A,W>A)))/float(sum(W_binary>A))
        else:
            prec_plus = 1

        if (sum(W_binary<A)):
            prec_minus = float(sum(np.multiply(W_binary<A,W<A)))/float(sum(W_binary<A))
        else:
            prec_minus = 1

        if (sum(W_binary==A)):
            prec_zero = float(sum(np.multiply(W_binary==A,W==A)))/float(sum(W_binary==A))
        else:
            prec_zero = 1
    #--------------------------------------------------------------------------
    
    #---------------------Reshape and Return the Results-----------------------
    recall = [acc_plus,acc_minus,acc_zero]
    precision = [prec_plus,prec_minus,prec_zero]
    return recall,precision
    #--------------------------------------------------------------------------
#==============================================================================    
