#==============================================================================
#================================NeuralNet Class===============================
# This class defines a multi-layer neural network with the given number of
# neurons and other properties. It also initializes the undefined variables
# with default values
#==============================================================================

from brian import*
import os

#-----------------------DEFAULT VALUES FOR THE VARIABLES-----------------------
N_EXC_ARRAY_DEFAULT = [160,1]
N_INH_ARRAY_DEFAULT = [40,0]
CONNECTION_PROB_DEFAULT = 0.2
NO_LAYERS_DEFAULT = 2
DELAY_MAX_DEFAULT = 1.0
RANDOM_DELAY_FLAG_DEFAULT = 0

#...........................THE DEFAULT NEURAL MODEL............................
tau=10*ms
tau_e=2*ms # AMPA synapse
eqs='''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''
NEURAL_MODEL_DEFAULT = list([eqs,tau,tau_e])
#...............................................................................
    
#.........................Connection Probability Matrix.........................
CONNECTION_PROB_MATRIX_DEFAULT = np.zeros([NO_LAYERS_DEFAULT,NO_LAYERS_DEFAULT])
for i in range(0,NO_LAYERS_DEFAULT):
    for j in range(i+1,NO_LAYERS_DEFAULT):
        l = j-i
        p_temp = CONNECTION_PROB_DEFAULT/(math.log(j) + 0.5772156 + 0.85/(2*j))   # Using Harmonic number approximations
        p_temp = 0.001*round(1000*p_temp/float(l))
        CONNECTION_PROB_MATRIX_DEFAULT[i,j] = p_temp
#...............................................................................
    
#...................Maximum Connection Delays Matrix............................
DELAY_MAX_MATRIX_DEFAULT = np.zeros([NO_LAYERS_DEFAULT,NO_LAYERS_DEFAULT])
for i in range(0,NO_LAYERS_DEFAULT):
    for j in range(i+1,NO_LAYERS_DEFAULT):
        DELAY_MAX_MATRIX_DEFAULT[i,j] = DELAY_MAX_DEFAULT*(0.9*(j-i))
#...............................................................................

#-------------------------------------------------------------------------------

class NeuralNet():
    
    #--------------------------INITIALIZE THE CLASS----------------------------
    def __init__(self,no_layers = None, n_exc_array = None, n_inh_array = None, connection_prob_matrix = None,delay_max_matrix = None,random_delay_flag = None,neural_model_eq=None):
        
        #................Assign Default Variables if Necessary.................
        if (no_layers is None):
            self.no_layers = NO_LAYERS_DEFAULT
            print('ATTENTION: The default value of %s for no_layers is considered.\n' %str(no_layers))
        else:
            self.no_layers = no_layers
        
        if (n_exc_array is None):
            self.n_exc_array = N_EXC_ARRAY_DEFAULT
            print('ATTENTION: The default value of %s for n_exc_array is considered.\n' %str(n_exc_array))
        else:
            self.n_exc_array = n_exc_array
        
        if (n_inh_array is None):
            self.n_inh_array = N_INH_ARRAY_DEFAULT
            print('ATTENTION: The default value of %s for n_inh_array is considered.\n' %str(n_inh_array))
        else:
            self.n_inh_array = n_inh_array
        
        if (connection_prob_matrix is None):
            self.connection_prob_matrix = CONNECTION_PROB_MATRIX_DEFAULT
            print('ATTENTION: The default value of %s for connection_prob_matrix is considered.\n' %str(connection_prob_matrix))
        else:
            self.connection_prob_matrix = connection_prob_matrix
            
        if (delay_max_matrix is None):
            self.delay_max_matrix = DELAY_MAX_MATRIX_DEFAULT
            print('ATTENTION: The default value of %s for delay_max_matrix is considered.\n' %str(delay_max_matrix))
        else:
            self.delay_max_matrix = delay_max_matrix
        
        if (random_delay_flag is None):
            self.random_delay_flag = RANDOM_DELAY_FLAG_DEFAULT
            print('ATTENTION: The default value of %s for random_delay_flag is considered.\n' %str(random_delay_flag))
        else:
            self.random_delay_flag = random_delay_flag
            
        if (neural_model_eq is None):
            self.neural_model_eq = NEURAL_MODEL_DEFAULT
            print('ATTENTION: The default value of %s for n_exc_array is considered.\n' %str(neural_model_eq))
        else:
            self.neural_model_eq = neural_model_eq
        #......................................................................    
        
        #.........Check the Consistensy of the Initialized Variables...........
        if (len(self.n_exc_array) != self.no_layers):
            print('ERROR: The number of excitatory neurons per layer does not match the number of layers')
            sys.exit()
            
        if (len(self.n_inh_array) != self.no_layers):
            print('ERROR: The number of inhibitory neurons per layer does not match the number of layers')
            sys.exit()
        
        s = self.connection_prob_matrix.shape
        if (s[0] != self.no_layers):
            print('ERROR: The structure of the connection probability matrix does not match the number of layers')
            sys.exit()
        
        s = self.delay_max_matrix.shape
        if (s[0] != self.no_layers):
            print('ERROR: The structure of the maximum delay matrix does not match the number of layers')
            sys.exit()
        #......................................................................
            
    #--------------------------------------------------------------------------
    
    
    #----------------------------READ THE WEIGHTS------------------------------
    def read_weights(self,ensemble_count,file_name_base_data):
        
        #.....................Initialize Other Variables......................
        Neural_Connections = {}
        
        no_samples_per_cascade = max(3.0,25*self.no_layers*np.max(self.delay_max_matrix)) # Number of samples that will be recorded
        running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds

        str_p = ''
        str_d = ''
        str_n_exc = ''
        str_n_inh = ''
        
        for i in range(0,self.no_layers):
            str_n_exc = str_n_exc + '_' + str(int(self.n_exc_array[i]))
            str_n_inh = str_n_inh + '_' + str(int(self.n_inh_array[i]))
            for j in range(i,self.no_layers):
                str_p = str_p + '_' + str(self.connection_prob_matrix[i,j])
                str_d = str_d + '_' + str(self.delay_max_matrix[i,j])
        #......................................................................
        
        #.....................Construct Prpoper File Names......................
        file_name_ending = "L_%s" %str(int(self.no_layers))
        file_name_ending = file_name_ending + "_n_exc" + str_n_exc
        file_name_ending = file_name_ending + "_n_inh" + str_n_inh
        file_name_ending = file_name_ending + "_p" + str_p 
        #file_name_ending = file_name_ending + "_q_%s" %str(frac_stimulated_neurons)
        file_name_ending = file_name_ending + "_R_%s" %str(self.random_delay_flag)    
        file_name_ending = file_name_ending + "_d" + str_d
        #file_name_ending = file_name_ending + "_T_%s" %str(no_stimul_rounds)    
        file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
        #......................................................................
              
        #....................Read the Weights and Delays.......................
        W_flag = 1
        for l_in in range(0,self.no_layers):
            n_exc_in = self.n_exc_array[l_in]
            n_inh_in = self.n_inh_array[l_in]
            n_in = n_exc_in + n_inh_in
            
            for l_out in range(l_in,self.no_layers):
                
                n_exc_out = self.n_exc_array[l_out]
                n_inh_out = self.n_inh_array[l_out]
                n_out = n_exc_out + n_inh_out
                file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                file_name = file_name_base_data + "/Graphs/We_%s.txt" %file_name_ending_temp
    
                if (os.path.isfile(file_name)):
                    if (os.stat(file_name)[6]):
                        We = np.genfromtxt(file_name, dtype=None, delimiter='\t')                    
                        We = We.reshape(n_exc_in,n_out)
                    else:
                        We = []
                else:
                    W_flag = 0
                    break
    
                file_name = file_name_base_data + "/Graphs/Wi_%s.txt" %file_name_ending_temp                
                if (os.path.isfile(file_name)):
                    if (os.stat(file_name)[6]):
                        Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                        Wi = Wi.reshape(n_inh_in,n_out)
                    else:
                        Wi = []
                else:
                    W_flag = 0
                    break
        
                file_name = file_name_base_data + "/Graphs/De_%s.txt" %file_name_ending_temp    
                if (os.path.isfile(file_name)):
                    if (os.stat(file_name)[6]):
                        De = np.genfromtxt(file_name, dtype=None, delimiter='\t')                    
                        De = De.reshape(n_exc_in,n_out)
                    else:
                        De = []
                else:
                    W_flag = 0
                    break
    
                file_name = file_name_base_data + "/Graphs/Di_%s.txt" %file_name_ending_temp
                if (os.path.isfile(file_name)):
                    if (os.stat(file_name)[6]):
                        Di = np.genfromtxt(file_name, dtype=None, delimiter='\t')                    
                        Di = Wi.reshape(n_inh_in,n_out)
                    else:
                        Di = []
                else:
                    W_flag = 0        
                    break
        
                delay_max = self.delay_max_matrix[l_in,l_out]
                ind = str(l_in) + str(l_out)
                if len(We) and len(Wi):
                    W = np.vstack([We,Wi])
                elif len(We):
                    W = We
                else:
                    W = Wi
                    
                if len(De) and len(Di):
                    D = np.vstack([De,Di])
                elif len(De):
                    D = De
                else:
                    D = Di
                Neural_Connections[ind] = list([W,D,delay_max])
            
            
            if (W_flag == 0):
                break
        #......................................................................
        
        if W_flag:
            Neural_Connections['n_exc'] = self.n_exc_array
            Neural_Connections['n_inh'] = self.n_inh_array
            self.Neural_Connections = Neural_Connections
        else:
            self.Neural_Connections = []
            
        return W_flag
    #--------------------------------------------------------------------------
    
    #----------------------------READ THE WEIGHTS------------------------------
    def create_weights(self,ensemble_count,file_name_base_data):
        
        #.....................Initialize Other Variables......................
        Neural_Connections = {}                
        no_samples_per_cascade = max(3.0,25*self.no_layers*np.max(self.delay_max_matrix)) # Number of samples that will be recorded
        running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds

        str_p = ''
        str_d = ''
        str_n_exc = ''
        str_n_inh = ''
        for i in range(0,self.no_layers):
            str_n_exc = str_n_exc + '_' + str(int(self.n_exc_array[i]))
            str_n_inh = str_n_inh + '_' + str(int(self.n_inh_array[i]))
            for j in range(i,self.no_layers):
                str_p = str_p + '_' + str(self.connection_prob_matrix[i,j])
                str_d = str_d + '_' + str(self.delay_max_matrix[i,j])
        #......................................................................
        
        #.....................Construct Prpoper File Names......................
        file_name_ending = "L_%s" %str(int(self.no_layers))
        file_name_ending = file_name_ending + "_n_exc" + str_n_exc
        file_name_ending = file_name_ending + "_n_inh" + str_n_inh
        file_name_ending = file_name_ending + "_p" + str_p 
        #file_name_ending = file_name_ending + "_q_%s" %str(frac_stimulated_neurons)
        file_name_ending = file_name_ending + "_R_%s" %str(self.random_delay_flag)    
        file_name_ending = file_name_ending + "_d" + str_d
        #file_name_ending = file_name_ending + "_T_%s" %str(no_stimul_rounds)    
        file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
        #......................................................................
              
        #........................Initialize the Layers.........................        
        neurons_exc = []
        neurons_inh = []
        neural_layers = []
        Neural_Connections = {}
        Neural_Connections['n_exc'] = self.n_exc_array
        Neural_Connections['n_inh'] = self.n_inh_array
        
        for l in range(0,self.no_layers):
        
            n_exc = int(self.n_exc_array[l])
            n_inh = int(self.n_inh_array[l])
            n = n_exc + n_inh
            neurons = NeuronGroup(n,model=self.neural_model_eq[0],threshold=10*mV,reset=0*mV,refractory=1*ms)
            
            Pe = neurons.subgroup(n_exc)
            Pi = neurons.subgroup(n_inh)
            neural_layers.append(neurons)
            neurons_exc.append(Pe)
            neurons_inh.append(Pi)        
        #......................................................................


        #.....................Connect the Layers Together......................
        for l_in in range(0,self.no_layers):
            Pe = neurons_exc[l_in]
            Pi = neurons_inh[l_in]
            n_exc = self.n_exc_array[l_in]
            n_inh = self.n_inh_array[l_in]

            
            for l_out in range(l_in,self.no_layers):
                output_layer = neural_layers[l_out]
                connection_prob = self.connection_prob_matrix[l_in,l_out]
                delay_max = self.delay_max_matrix[l_in,l_out]
                
                if self.random_delay_flag:                    
                    Ce = Connection(Pe, output_layer, weight=1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * abs(sign(j-i))*rand(1) * ms)
                    Ci = Connection(Pi, output_layer, weight=-1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * abs(sign(j-i))* rand(1) * ms)                    
                else:
                    Ce = Connection(Pe, output_layer, weight=1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * abs(sign(j-i)) * ms)
                    Ci = Connection(Pi, output_layer, weight=-1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * abs(sign(j-i)) * ms)
            
                #............Transform Connections to Weighted Matrices............
                We = Ce.W.todense()
                Wi = Ci.W.todense()
                De = Ce.delay.todense()
                Di = Ci.delay.todense()
                
                W = np.vstack([We,Wi])
                D = np.vstack([De,Di])
                
                ind = str(l_in) + str(l_out)
                Neural_Connections[ind] = list([W,D,delay_max])
                #..................................................................
                
                #..............Save Connectivity Matrices to the File..............
                file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                file_name = file_name_base_data + "/Graphs/We_%s.txt" %file_name_ending_temp
                np.savetxt(file_name,We,'%1.4f',delimiter='\t',newline='\n')
                file_name = file_name_base_data + "/Graphs/Wi_%s.txt" %file_name_ending_temp
                np.savetxt(file_name,Wi,'%1.4f',delimiter='\t',newline='\n')
        
                file_name = file_name_base_data + "/Graphs/De_%s.txt" %file_name_ending_temp
                np.savetxt(file_name,De,'%1.4f',delimiter='\t',newline='\n')
                file_name = file_name_base_data + "/Graphs/Di_%s.txt" %file_name_ending_temp
                np.savetxt(file_name,Di,'%1.4f',delimiter='\t',newline='\n')
                #..................................................................

        #......................................................................
        
        Neural_Connections['n_exc'] = self.n_exc_array
        Neural_Connections['n_inh'] = self.n_inh_array
        self.Neural_Connections = Neural_Connections
        return 1 #Neural_Connections
    #--------------------------------------------------------------------------
    
    
    
    
    def junk(self):
        #----------------Check for and Apply the Triangle Inequality---------------
        if random_delay_flag:
            for l_in in range(0,no_layers):                        
                for l_out in range(l_in+1,no_layers):
                    # The following lines of code need serious rethinking as we do not take into account the three-hop, 4-hop, etc.
                    # trajectories to reach a neuron. The probability of this event might be rare but if we plan not to ignore it,
                    # we'd better modify the code. Or, just consider neighboring layers. 
                    file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out-1)
                
                    file_name = file_name_base_data + "/Graphs/De_%s.txt" %file_name_ending_temp
                    De = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(De.shape) == 1):
                        De = De.reshape(n_exc,1)
                
                    file_name = file_name_base_data + "/Graphs/Di_%s.txt" %file_name_ending_temp
                    Di = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(Di.shape) == 1):
                        Di = Di.reshape(n_inh,1)
                    
                    D_lay_1 = np.vstack((De,Di))
                
                
                    file_name_ending_temp = file_name_ending + '_l_' + str(l_in+1) + '_to_' + str(l_out)
                    
                    file_name = file_name_base_data + "/Graphs/De_%s.txt" %file_name_ending_temp
                    De = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(De.shape) == 1):
                        De = De.reshape(n_exc,1)        
                
                    file_name = file_name_base_data + "/Graphs/Di_%s.txt" %file_name_ending_temp
                    Di = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(Di.shape) == 1):
                        Di = Di.reshape(n_inh,1)
                
                    D_lay_2 = np.vstack((De,Di))
                
                
                    file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                    
                    file_name = file_name_base_data + "/Graphs/De_%s.txt" %file_name_ending_temp
                    De = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(De.shape) == 1):
                        De = De.reshape(n_exc,1)
                
                    file_name = file_name_base_data + "/Graphs/Di_%s.txt" %file_name_ending_temp
                    Di = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(Di.shape) == 1):
                        Di = Di.reshape(n_inh,1)
                
                    D_lay_1_2 = np.vstack((De,Di))
                
                    D_temp = np.dot(D_lay_1,D_lay_2)
                    temp = np.multiply(D_temp,D_lay_1_2)
                    
                    ind = np.nonzero(temp)
                    ind = np.transpose(ind)
                    lll = ind.shape
                    lll = lll[0]
                
                    for k in range(0,lll):
                        row = ind[k,0]
                        col = ind[k,1]
                    
                        temp2 = D_lay_1[row,:]
                        ind2 = np.nonzero(temp2)
                        ind2 = np.transpose(ind2)
                    
                        lll2 = len(ind2)
                    
                    
                        for kk in range(0,lll2):
                            d1 = D_lay_1[row,ind2[kk]]
                            d2 = D_lay_2[ind2[kk],col]
                            d_long = D_lay_1_2[row,col]
                            if (d2):
                                if ( (d1+d2) < d_long):
                                    D_lay_1_2[row,col] = d1 + d2 + 0.000005
                                    print d1,d2,d_long
                
                    
                        #......................Save the Results........................
                        n_exc = n_exc_array[l_in]
                        n_inh = n_inh_array[l_in]
                        De = D_lay_1_2[n_exc,:]
                        Di = D_lay_1_2[n_exc+1:n_exc+n_inh,:]
                        
                        file_name = file_name_base_data + "/Graphs/De_%s.txt" %file_name_ending_temp
                        np.savetxt(file_name,De,'%1.4f',delimiter='\t',newline='\n')
                        file_name = file_name_base_data + "/Graphs/Di_%s.txt" %file_name_ending_temp
                        np.savetxt(file_name,Di,'%1.4f',delimiter='\t',newline='\n')
                        #..........................................................
        
        #--------------------------------------------------------------------------