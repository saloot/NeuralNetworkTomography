import math
from brian import *
from scipy import sparse

#=======DETERMINING PROPER THRESHOLD BINARIFICATION OF INFERRED GRAPHS=========
def q_func_scalar(x):
    import math
    return 0.5 - 0.5 * math.erf(x/math.sqrt(2))
#==============================================================================        


#=======DETERMINING PROPER THRESHOLD BINARIFICATION OF INFERRED GRAPHS=========
def determine_binary_threshold(n,p_plus,p_minus,theta,T,Delta,q):
    from math import sqrt    
    
    no_input_ones = q * n
    mu = (p_plus - p_minus) * no_input_ones
    var = math.sqrt((p_plus * (1 - p_plus) - p_minus * (1-p_minus) + 2 * p_plus * p_minus) * no_input_ones)
    mu_p = (p_plus - p_minus) * (no_input_ones - 1)
    var_p = math.sqrt((p_plus * (1 - p_plus) - p_minus * (1-p_minus) + 2 * p_plus * p_minus) * (no_input_ones - 1) )

    p_W_plus_1 = q_func_scalar((theta - mu_p - 1)/var_p)
    p_W_zero_1 = q_func_scalar((theta - mu_p)/var_p)
    p_W_minus_1 = q_func_scalar((theta - mu_p + 1)/var_p)
                
                
    p_W_plus_0 = 1 - p_W_plus_1
    p_W_zero_0 = 1 - p_W_zero_1
    p_W_minus_0 = 1 - p_W_minus_1
                                
    e1 = p_W_plus_1 - p_W_plus_0
    e2 = p_W_minus_1 - p_W_minus_0
    epsilon_plus_base = e1# - (e1-e2)/4
    epsilon_minus_base = e2# + (e1-e2)/4
    print e1
    print epsilon_plus_base
    
    epsilon_plus = epsilon_plus_base * Delta * T * q
    epsilon_minus = epsilon_minus_base * Delta * T * q
    
    return (epsilon_plus,epsilon_minus)
#==============================================================================        
    

#============================GENERATE FIRING RATES=============================
def myrates(t):
    rates=zeros(n)*Hz    
    if t < 0.1 * ms:
        input_index = floor(n*rand(round(n*qqi)))
        input_index = input_index.astype(int)
        rates[input_index]=ones(round(n*qqi))*input_stimulus_freq *Hz
    return rates
#==============================================================================



#=================GENERATE NEURAL ACTIVITY FOR THE DELAYED NETWORK=============
def generate_activity_recurrent(n_exc,n_inh,running_period,We,Wi,De,Di,S_time_file,eqs,tau,tau_e,delay_max,frac_input_neurons):
    
    #--------------------------Initializing Variables--------------------------
    global n
    global qqi
    global input_stimulus_freq
    n = n_exc+n_inh
    qqi = frac_input_neurons
    input_stimulus_freq = 20000               # The frequency of spikes by the neurons in the input layer (in Hz)
    #--------------------------------------------------------------------------
    
    #----------------------Initialize the Main Layer---------------------------
    main_network=NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)
    Pe = main_network.subgroup(n_exc)
    Pi = main_network.subgroup(n_inh)
    h = float(n_exc)/float(2)    
    
    Ce = DelayConnection(Pe, main_network,max_delay = delay_max*ms,delay = lambda i, j:delay_max * ms * abs(i /h - 1) ** 2)
    
    h = float(n_inh)/float(2)
    Ci = DelayConnection(Pi, main_network,max_delay = delay_max*ms,delay = lambda i, j:delay_max * ms * abs(i /h - 1) ** 2)
    
    Ce.connect(Pe,main_network,sparse.csc_matrix(We))    
    Ci.connect(Pi,main_network,sparse.csc_matrix(Wi))
    
    Ce.set_delays(Pe,main_network,sparse.csc_matrix(De))
    Ci.set_delays(Pi,main_network,sparse.csc_matrix(Di))
    #Ce.connect_from_sparse(sparse.csc_matrix(We),sparse.csc_matrix(De))
    #Ci.connect_from_sparse(sparse.csc_matrix(Wi),sparse.csc_matrix(Di))
    
    net = Network(main_network,[Ce,Ci]) 
    #--------------------------------------------------------------------------
        
    #-------------------Initialize the Input Stimulus----------------------        
    inputer_dummy_layer=PoissonGroup(n,myrates)
    input_connections=Connection(inputer_dummy_layer,main_network,weight=lambda i,j:(1-abs(sign(i-j))),delay = 0*ms)
    #----------------------------------------------------------------------
    net.add(input_connections)
    net.add(inputer_dummy_layer)
        
    M_l1 = SpikeMonitor(inputer_dummy_layer)
    M_l2 = SpikeMonitor(main_network)
    net.add(M_l1)
    net.add(M_l2)
    net.run(running_period * ms)        
    
    print M_l1.nspikes, "spikes in input layer"
    print M_l2.nspikes, "spikes in output layer"
    #----------------------------------------------------------------------
    
    #--------------------Save Spike Times to the File----------------------        
    SS = M_l2.spikes
    for l in range(0,len(SS)):
        item = SS[l]
        a = item[0]
        b = item[1]
        b = b.astype(float)
        S_time_file.write("%d \t %f \n" %(a,b))
        #S_time_file.write("\n")
    
    S_time_file.write("-2 \t -2 \n")
    #----------------------------------------------------------------------    
    M_l1.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0
    M_l2.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0
    
    WWe = Ce.W.todense()
    WWi = Ci.W.todense()
    DDe = Ce.delay.todense()
    DDi = Ci.delay.todense() 
    
    #clear(net)
    clear(spikequeue)
    
    return (WWe,WWi,DDe,DDi)
#==============================================================================



#============GENERATE NEURAL ACTIVITY FOR THE FEEDFORWARD NETWORK==============
def generate_activity_FF_n_1(n_exc,n_inh,running_period,We,Wi,De,Di,S_time_file,S_time_file_out,eqs,tau,tau_e,delay_max,frac_input_neurons,cascade_count):
    
    #--------------------------Initializing Variables--------------------------
    global n
    global qqi
    global input_stimulus_freq
    n = n_exc+n_inh
    qqi = frac_input_neurons
    input_stimulus_freq = 20000               # The frequency of spikes by the neurons in the input layer (in Hz)
    #--------------------------------------------------------------------------
    
    #----------------------Initialize the Main Layer---------------------------    
    main_network=NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)
    output_neuron = NeuronGroup(1,model=eqs,threshold=10*mV,reset=0*mV)
    
    
    Pe = main_network.subgroup(n_exc)
    Pi = main_network.subgroup(n_inh)
    h = float(n_exc)/float(2)
        
    Ce = DelayConnection(Pe, output_neuron,max_delay = delay_max*ms,delay = lambda i, j:delay_max * ms * abs(i /h - 1) ** 2)
    Ce.connect(Pe,output_neuron,sparse.csc_matrix(We))
    Ce.set_delays(Pe,output_neuron,sparse.csc_matrix(De))
    
    h = float(n_inh)/float(2)
    
    
    Ci = DelayConnection(Pi, output_neuron,max_delay = delay_max*ms,delay = lambda i, j:delay_max * ms * abs(i /h - 1) ** 2)
    Ci.connect(Pi,output_neuron,sparse.csc_matrix(Wi))
    Ci.set_delays(Pi,output_neuron,sparse.csc_matrix(Di))

    net = Network(main_network,[Ce,Ci]) 
    #--------------------------------------------------------------------------
        
    #-------------------Initialize the Input Stimulus----------------------        
    inputer_dummy_layer=PoissonGroup(n,myrates)
    input_connections=Connection(inputer_dummy_layer,main_network,weight=lambda i,j:(1-abs(sign(i-j))),delay = 0*ms)
    #----------------------------------------------------------------------
        
    net.add(input_connections)
    net.add(inputer_dummy_layer)
    net.add(output_neuron)
        
    M_l1 = SpikeMonitor(inputer_dummy_layer)
    M_l2 = SpikeMonitor(main_network)
    M_l3 = SpikeMonitor(output_neuron)        
    net.add(M_l1)
    net.add(M_l2)
    net.add(M_l3)
    net.run(running_period * ms)        
        
    print M_l1.nspikes, "spikes in dummy layer"
    print M_l2.nspikes, "spikes in middle layer"
    print M_l3.nspikes, "spikes in output layer"
    #----------------------------------------------------------------------
    
    
    
    #--------------------Save Spike Times to the File----------------------        
    SS = M_l2.spikes
    for l in range(0,len(SS)):
        item = SS[l]
        a = item[0]
        b = item[1]
        b = b.astype(float)
        S_time_file.write("%d \t %f \n" %(a,b))
        #S_time_file.write("\n")
    
    S_time_file.write("-2 \t -2 \n")
    
    
    SS = M_l3.spiketimes
    
    if (SS[0]>0):
        b = SS[0]
        b = b[0]
        b = b.astype(float)        
        S_time_file_out.write("%d \t %f \n" %(cascade_count,b))
        #S_time_file_out.write("\n")
    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------    
    M_l1.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0
    M_l2.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0
    M_l3.source.clock.reinit()                  # Reset the spikemonitor's clock so that for the next random network, everything starts from t=0                        
    WWe = Ce.W.todense()
    WWi = Ci.W.todense()
    DDe = Ce.delay.todense()
    DDi = Ci.delay.todense()
    
    clear(spikequeue)
    #clear(net)
    
    
    return (WWe,WWi,DDe,DDi)
#==============================================================================




#==============================================================================
def beliefs_to_binary(binary_mode,W_inferred,n,p_plus,p_minus,thea,T,Delta,params,compensate_flag):
    import numpy as np
    connection_prob = p_plus + p_minus
    a = W_inferred.shape    
    n = a[0]
    if (len(a)>1):
        W_binary = np.zeros([n,n])
        W_binary.fill(0)
    else:
        W_binary = np.zeros([n])
        W_binary.fill(0)
    if (binary_mode == 1):
        q = params[0]
        r = params[1]
        epsilon_plus,epsilon_minus = determine_binary_threshold(n,p_plus,p_minus,theta,T,Delta,q)
            
        #epsilon_plus_adjusted = epsilon_plus * q/frac_input_neurons
        #epsilon_minus_adjusted = epsilon_minus * q/frac_input_neurons
            
        epsilon_plus_adjusted = epsilon_plus  * (2*r-1)/(2*p_W_zero_1-1)
        epsilon_minus_adjusted = epsilon_minus * (2*r-1)/(2*p_W_zero_1-1)

        temp = (W_inferred > epsilon_plus_adjusted)
        temp = temp.astype(int)            
        W_binary = W_binary + temp
            
        temp = (W_inferred < epsilon_minus_adjusted)
        temp = temp.astype(int)
        W_binary = W_binary - temp
    elif (binary_mode == 2):
        
        if (len(a)>1):
            for i in range(0,n):
                w_temp = W_inferred[:,i]
                w_temp_s = np.sort(W_inferred[:,i])
                ind = np.argsort(w_temp)
                w_temp = np.zeros(n)
                w_temp[ind[0:int(round(p_minus*n))+1]] = -1            
                w_temp[ind[len(ind)-int(round(p_plus*n))+1:len(ind)+1]] = 1            
                W_binary[:,i] = w_temp
        else:
            ind = np.argsort(W_inferred)
            w_temp = np.zeros([n])
            w_temp[ind[0:int(round(p_minus*n))+1]] = -1
            w_temp[ind[len(ind)-int(round(p_plus*n))+1:len(ind)+1]] = 1            
            W_binary = w_temp
        
    elif (binary_mode == 3):
        
        #---Process to make sure that all outgoing links have the same type----
        total_belief = np.sum(W_inferred,axis=1)
        temp1 = np.sort(total_belief)
        ind = np.argsort(total_belief)
        
        
        for i in ind[len(ind)-int(round(p_plus*n))+1:len(ind)+1]:
            w_temp = W_inferred[i,:]
            w_temp_s = np.sort(W_inferred[i,:])
            ind2 = np.argsort(w_temp)
            w_temp = np.zeros(n) 
            w_temp[ind2[len(ind2)-int(round(connection_prob*n))+1:len(ind2)+1]] = 1
            W_binary[i,:] = w_temp
            
        for i in ind[0:int(round(p_minus*n))+1]:
            w_temp = W_inferred[i,:]
            w_temp_s = np.sort(W_inferred[i,:])
            ind2 = np.argsort(w_temp)
            w_temp = np.zeros(n) 
            w_temp[ind2[0:int(round(connection_prob*n))+1]] = -1
            W_binary[i,:] = w_temp
        #----------------------------------------------------------------------
        
        
    if compensate_flag:
        #---Process to make sure that all outgoing links have the same type----
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
        #----------------------------------------------------------------------
        
    return W_binary
#==============================================================================


def calucate_accuracy(W_binary,W):
    a = W.shape    
    n = a[0]
    
    if (len(a)>1):
        acc_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W>np.zeros([n,n]))))
        acc_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W<np.zeros([n,n]))))
        acc_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W==np.zeros([n,n]))))
        
        prec_plus = float(sum(sum(np.multiply(W_binary>np.zeros([n,n]),W>np.zeros([n,n])))))/float(sum(sum(W_binary>np.zeros([n,n]))))
        prec_minus = float(sum(sum(np.multiply(W_binary<np.zeros([n,n]),W<np.zeros([n,n])))))/float(sum(sum(W_binary<np.zeros([n,n]))))
        prec_zero = float(sum(sum(np.multiply(W_binary==np.zeros([n,n]),W==np.zeros([n,n])))))/float(sum(sum(W_binary==np.zeros([n,n]))))
    
    recall = [acc_plus,acc_minus,acc_zero]
    precision = [prec_plus,prec_minus,prec_zero]
    return recall,precision
#==============================================================================    