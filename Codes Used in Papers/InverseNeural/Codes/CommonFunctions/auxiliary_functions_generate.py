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

# INPUTS:
#    NeuralNetwork: the "Network" object, containing key information about the network
#    running_period: the duration of generated data, in miliseconds
#    S_time_file_base: the base name for the files where the data should be stored.The actual file name 
#                      will be this base name plus some extensions for each layer.
#    frac_input_neurons: fraction (or probability) of neurons being stimulated (in feed-forward networks with stimulate-and-observe strategy)
#                        or probability of being trigerred by external traffic (in recurrent networks or feed-forward networks without stimulate-and-observe strategy)
#    generate_data_mode: 'F' for stimulate-and-observe scenario
#                        'R' for the general case, where all neurons in the network can be trigerred due to external traffic  

# OUTPUTS:
#    Neural_Connections_Out: the object that contains the information about the connection weights and delays for the networks (used for control purposes)
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
        
        #..................Generate Random 'External' Traffic..................
        for j in range(0,n):
            #no_stim_points = int(running_period/3.2)
            no_stim_points = int(frac_input_neurons* running_period*(0.5+rand(1)/1.0))
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
        #........................................................................
        
    else:
        
        #....................Stimulate-and-Observe Scenario......................
        no_stimulated_neurons = int(frac_input_neurons * n)
        temp = range(0,int(n))
        random.shuffle(temp)
        neuron_indices.extend(temp[0:no_stimulated_neurons])
        stimul_times.extend(0.1 *np.ones([no_stimulated_neurons]))
        for i in range(0,no_stimulated_neurons):
            stimul_times[i] = stimul_times[i] *ms        
        #neuron_indices = 
        #........................................................................
        
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
    
    


#==============================================================================
#===========================verify_neural_activity=============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function runs the neural networks and generatethe required neural
# activity. The Brian simulator is used for this part.

# INPUTS:
#    NeuralNetwork: the "Network" object, containing key information about the network
#    Network_in: the connectivity information for the inferred graph
#    running_period: the duration of generated data, in miliseconds
#    S_time_file_base: the base name for the files where the data should be stored.The actual file name 
#                      will be this base name plus some extensions for each layer.
#    frac_input_neurons: fraction (or probability) of neurons being stimulated (in feed-forward networks with stimulate-and-observe strategy)
#                        or probability of being trigerred by external traffic (in recurrent networks or feed-forward networks without stimulate-and-observe strategy)
#    stimulated_neurons: the index of stimulated neurons in each round
#    stimulation_times: the time instances where each neuron is stimulated

# OUTPUTS:
#    Neural_Connections_Out: the object that contains the information about the connection weights and delays for the networks (used for control purposes)
#    out_spike: the reproduces spikes
#------------------------------------------------------------------------------

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


