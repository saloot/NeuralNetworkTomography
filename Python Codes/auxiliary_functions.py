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
    SS = M_l2.spiketimes
    for l in range(0,len(SS)):
        item = SS[l]
        if (len(item) == 0):
            S_time_file.write("0 \t")
        else:
            for j in range(0,len(item)):
                if (item[j]<0.00015):
                    S_time_file.write("%f \t" % item[j])
                elif (item[j]>0.00015):
                    S_time_file.write("%f \t" % item[j])
                    break                
        S_time_file.write("-1 \n")
    
        
    S_time_file.write("-2 \n")
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
def generate_activity_FF_n_1(n_exc,n_inh,running_period,We,Wi,De,Di,S_time_file,S_time_file_out,eqs,tau,tau_e,delay_max,frac_input_neurons):
    
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
    SS = M_l2.spiketimes
    for l in range(0,len(SS)):
        item = SS[l]
        if (len(item) == 0):
            S_time_file.write("0 \t")
        else:
            for j in range(0,len(item)):
                if (item[j]>0.000001):
                    S_time_file.write("%f \t" % item[j])
        S_time_file.write("-1 \n")
        
    SS = M_l3.spiketimes
    for l in range(0,len(SS)):
        item = SS[l]
        if (len(item) == 0):
            S_time_file_out.write("0 \t")
        else:
            for j in range(0,len(item)):
                if (item[j]>0.000):
                    S_time_file_out.write("%f \t" % item[j])
                    break                
    
    S_time_file_out.write("-2 \n")
    S_time_file.write("-2 \n")

        
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