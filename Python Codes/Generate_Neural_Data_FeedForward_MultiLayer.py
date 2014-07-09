#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.2
frac_input_neurons_default = 0.4
no_cascades_default = 50
ensemble_size_default = 1
delay_max_default = 1.0
file_name_base_data_default = "./Data/MultiLayerFeedForward"
ensemble_count_init_default = 0
no_layers_default = 1
random_delay_flag_default = 0
#==============================================================================

#================================INSTRUCTIONS==================================
help_message = "\n"
help_message = help_message + "\n"
help_message = help_message + "###################################INSTRUCTIONS################################\n"
help_message = help_message + "Here is how to use the code: you have to specify the option flag and"
help_message = help_message + "the quantity right afterwards.\nExample: -E 100 for setting a network with 100 excitatory neurons. "
help_message = help_message + "The full list of options are as follows:\n"
help_message = help_message + "-E xxx: To specify the number of excitatory neurons. Default value = %s.\n" %str(n_exc_default)
help_message = help_message + "-I xxx: To specify the number of inhibitory neurons. Default value = %s.\n" %str(n_inh_default)
help_message = help_message + "-P xxx: To specify the probabaility of having a connection between two neurons. Default value = %s.\n" %str(connection_prob_default)
help_message = help_message + "-Q xxx: To specify the fraction of stimulated input neurons. Default value = %s.\n" %str(frac_input_neurons_default)
help_message = help_message + "-T xxx: To specify the number of considered cascades. Default value = %s.\n" %str(no_cascades_default)
help_message = help_message + "-D xxx: To specify the maximum delay for the neural connections in milliseconds. Default value = %s.\n" %str(delay_max_default)
help_message = help_message + "-S xxx: To specify the number of generated random graphs. Default value = %s.\n" %str(ensemble_size_default)
help_message = help_message + "-A xxx: To specify the folder that stores the generated data. Default value = %s. \n" %str(file_name_base_data_default)
help_message = help_message + "-F xxx: To specify the ensemble index to start simulation. Default value = %s. \n" %str(ensemble_count_init_default)
help_message = help_message + "-L xxx: To specify the number of layers in the network. Default value = %s. \n" %str(no_layers_default)
help_message = help_message + "-R xxx: To specify if the delays are fixed (R=0) or random (R=1). Default value = %s. \n" %str(random_delay_flag_default)
help_message = help_message + "#################################################################################"
help_message = help_message + "\n"
#==============================================================================


#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import time
import numpy as np
import os
import sys,getopt,os
import auxiliary_functions
reload(auxiliary_functions)
from auxiliary_functions import generate_neural_activity

%load_ext autoreload
%reload_ext autoreload
%reload_ext brian
from brian import *
#os.chdir('C:\Python27')
#os.chdir('/home/salavati/Desktop/Neural_Tomography')
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
os.system('clear')                                              # Clear the commandline window
t0 = time.time()                                                     # Initialize the timer


input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:B:D:A:F:R:")
if (input_opts):
    for opt, arg in input_opts:
        if opt == '-E':
            n_exc = int(arg)                                    # The number of excitatory neurons in the output layer
        elif opt == '-I':
                n_inh = int(arg)                                # The number of inhibitory neurons in the output layer
        elif opt == '-P':
            connection_prob = float(arg)                        # The probability of having a link from the input to output neurons in the second layer
        elif opt == '-Q':
            frac_input_neurons = float(arg)                     # Fraction of neurons in the input layer that will be excited by a stimulus
        elif opt == '-T':
            no_cascades = int(arg)                              # Number of times we inject stimulus to the network
        elif opt == '-S':
            ensemble_size = int(arg)                            # The number of random networks that will be generated
        elif opt == '-B':
            binary_mode = int(arg)                              # Defines the method to transform the graph to binary. "1" for threshold base and "2" for sparsity based
        elif opt == '-D':
            delay_max = float(arg)                              # The maximum amount of synaptic delay in mili seconds
        elif opt == '-A':
            file_name_base_data = str(arg)                      # The folder to store results
        elif opt == '-F':
            ensemble_count_init = int(arg)                      # The ensemble to start simulations from
        elif opt == '-R':
            random_delay_flag = int(arg)                             # The ensemble to start simulations from            
        elif opt == '-h':
            print(help_message)
            sys.exit()
else:
    print('Code will be executed using default values')
 
#==============================================================================


#================================INITIALIZATIONS===============================

#------------Set the Default Values if Variables are not Defines---------------
if 'frac_input_neurons' not in locals():
    frac_input_neurons = frac_input_neurons_default
    print('ATTENTION: The default value of %s for frac_input_neurons is considered.\n' %str(frac_input_neurons))

if 'no_cascades' not in locals():        
    no_cascades = no_cascades_default
    print('ATTENTION: The default value of %s for no_cascades is considered.\n' %str(no_cascades))

if 'ensemble_size' not in locals():            
    ensemble_size = ensemble_size_default
    print('ATTENTION: The default value of %s for ensemble_size is considered.\n' %str(ensemble_size))
    
if 'file_name_base_data' not in locals():
    file_name_base_data = file_name_base_data_default;
    print('ATTENTION: The default value of %s for file_name_base_data is considered.\n' %str(file_name_base_data))

if 'ensemble_count_init' not in locals():
    ensemble_count_init = ensemble_count_init_default;
    print('ATTENTION: The default value of %s for ensemble_count_init is considered.\n' %str(ensemble_count_init))

if 'no_layers' not in locals():
    no_layers = no_layers_default;
    print('ATTENTION: The default value of %s for no_layers is considered.\n' %str(no_layers))

if 'random_delay_flag' not in locals():
    random_delay_flag = random_delay_flag_default;
    print('ATTENTION: The default value of %s for random_delay_flag is considered.\n' %str(random_delay_flag))

if 'connection_prob' not in locals():
    connection_prob_matrix = np.zeros([no_layers,no_layers])
    for i in range(0,no_layers):
        for j in range(i,no_layers):
            connection_prob_matrix[i,j] = connection_prob_default/float(j-i+1)
    
    print('ATTENTION: The default value of %s for connection_prob is considered.\n' %str(connection_prob_matrix))
    
    
if 'delay_max' not in locals():
    delay_max_matrix = np.zeros([no_layers,no_layers])
    for i in range(0,no_layers):
        for j in range(i,no_layers):
            delay_max_matrix[i,j] = delay_max_default*(0.9*(j-i)+1)
            
    print('ATTENTION: The default value of %s for delay_max is considered.\n' %str(delay_max_matrix))

if 'n_exc_array' not in locals():
    n_exc_array = n_exc_default*np.ones([no_layers])
    print('ATTENTION: The default value of %s for n_exc is considered.\n' %str(n_exc_array))

if 'n_inh_array' not in locals():
    n_inh_array = n_inh_default*np.ones([no_layers])
    print('ATTENTION: The default value of %s for n_inh is considered.\n' %str(n_inh_array))
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if NEcessary---------------
if not os.path.isdir(file_name_base_data):
    os.makedirs(file_name_base_data)

temp = file_name_base_data + '/Graphs'
if not os.path.isdir(temp):    
    os.makedirs(temp)

temp = file_name_base_data + '/Spikes'
if not os.path.isdir(temp):        
    os.makedirs(temp)
#------------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
no_samples_per_cascade = max(3.0,25*np.max(delay_max_matrix)) # Number of samples that will be recorded
running_period = (no_samples_per_cascade/10.0)  # Total running time in mili seconds


str_p = ''
str_d = ''
str_n_exc = ''
str_n_inh = ''
for i in range(0,no_layers):
    str_n_exc = str_n_exc + '_' + str(int(n_exc_array[i]))
    str_n_inh = str_n_inh + '_' + str(int(n_inh_array[i]))
    for j in range(i,no_layers):
        str_p = str_p + '_' + str(connection_prob_matrix[i,j])
        str_d = str_d + '_' + str(delay_max_matrix[i,j])
#------------------------------------------------------------------------------

#----------------------------------Neural Model--------------------------------
tau=10*ms
tau_e=2*ms # AMPA synapse
eqs='''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''

neural_model_eq = list([eqs,tau,tau_e])
#------------------------------------------------------------------------------

#==============================================================================




for ensemble_count in range(ensemble_count_init,ensemble_size):

#============================GENERATE THE NETWORK==============================
        
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "L_%s" %str(int(no_layers))
    file_name_ending = file_name_ending + "_n_exc" + str_n_exc
    file_name_ending = file_name_ending + "_n_inh" + str_n_inh
    file_name_ending = file_name_ending + "_p" + str_p 
    file_name_ending = file_name_ending + "_q_%s" %str(frac_input_neurons)
    file_name_ending = file_name_ending + "_R_%s" %str(random_delay_flag)    
    file_name_ending = file_name_ending + "_d" + str_d
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #--------------------------------------------------------------------------
    
    file_name = file_name_base_data + "/Graphs/We_MLFF_n_1_%s.txt" %file_name_ending
    W_flag = 1
    if (os.path.isfile(file_name)):
        We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        We = We.reshape(n_exc,1)        
    else:
        W_flag = 0
    
    file_name = file_name_base_data + "/Graphs/Wi_MLFF_n_1_%s.txt" %file_name_ending
    if (os.path.isfile(file_name)):        
        Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        Wi = Wi.reshape(n_inh,1)        
    else:
        W_flag = 0
        
    file_name = file_name_base_data + "/Graphs/De_MLFF_n_1_%s.txt" %file_name_ending
    W_flag = 1
    if (os.path.isfile(file_name)):
        De = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        De = De.reshape(n_exc,1)        
    else:
        W_flag = 0
    
    file_name = file_name_base_data + "/Graphs/Di_MLFF_n_1_%s.txt" %file_name_ending
    if (os.path.isfile(file_name)):        
        Di = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        Di = Wi.reshape(n_inh,1)        
    else:
        W_flag = 0        
    
    if not W_flag:
        
        #---------------------------Initialize the Layers--------------------------
        neural_layers = []
        neurons_exc = []
        neurons_inh = []
        Neural_Connections = {}
        Neural_Connections['n_exc'] = n_exc_array
        Neural_Connections['n_inh'] = n_inh_array
        
        for l in range(0,no_layers):
        
            n_exc = int(n_exc_array[l])
            n_inh = int(n_inh_array[l])
            n = n_exc + n_inh
            neurons = NeuronGroup(n,model=eqs,threshold=10*mV,reset=0*mV)
            
            Pe = neurons.subgroup(n_exc)
            Pi = neurons.subgroup(n_inh)
            neural_layers.append(neurons)
            neurons_exc.append(Pe)
            neurons_inh.append(Pi)
            
        output_neuron = NeuronGroup(1,model=eqs,threshold=10*mV,reset=0*mV)
        neural_layers.append(output_neuron)
        #--------------------------------------------------------------------------
        
        #-----------------------Connect the Layers Together------------------------
        
        for l_in in range(0,no_layers):
            Pe = neurons_exc[l_in]
            Pi = neurons_inh[l_in]
            n_exc = n_exc_array[l_in]
            n_inh = n_inh_array[l_in]

            
            for l_out in range(l_in,no_layers):
                output_layer = neural_layers[l_out+1]
                connection_prob = connection_prob_matrix[l_in,l_out]
                delay_max = delay_max_matrix[l_in,l_out]
                
                if random_delay_flag:
                    Ce = Connection(Pe, output_layer, weight=1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * rand(1) * ms)
                    Ci = Connection(Pi, output_layer, weight=-1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * rand(1) * ms)
                else:
                    Ce = Connection(Pe, output_layer, weight=1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * ms)
                    Ci = Connection(Pi, output_layer, weight=-1*mV, sparseness=connection_prob,max_delay=delay_max * ms,delay=lambda i, j:delay_max * ms)
            
                #............Transform Connections to Weighted Matrices............
                We = Ce.W.todense()
                Wi = Ci.W.todense()
                De = Ce.delay.todense()
                Di = Ci.delay.todense()
                
                ind = str(l_in) + str(l_out)
                Neural_Connections[ind] = list([We,De,Wi,Di,delay_max])
                #..................................................................
                
                #..............Save Connectivity Matrices to the File..............
                file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out-1)
                file_name = file_name_base_data + "/Graphs/We_MLFF_%s.txt" %file_name_ending_temp
                np.savetxt(file_name,We,'%1.4f',delimiter='\t',newline='\n')
                file_name = file_name_base_data + "/Graphs/Wi_MLFF_%s.txt" %file_name_ending_temp
                np.savetxt(file_name,Wi,'%1.4f',delimiter='\t',newline='\n')
        
                file_name = file_name_base_data + "/Graphs/De_MLFF_%s.txt" %file_name_ending_temp
                np.savetxt(file_name,De,'%1.4f',delimiter='\t',newline='\n')
                file_name = file_name_base_data + "/Graphs/Di_MLFF_%s.txt" %file_name_ending_temp
                np.savetxt(file_name,Di,'%1.4f',delimiter='\t',newline='\n')
                #..................................................................

        #--------------------------------------------------------------------------    
    

        #----------------Check for and Apply the Triangle Inequality---------------
        if random_delay_flag:
            for l_in in range(0,no_layers):                        
                for l_out in range(l_in+1,no_layers):
                    # The following lines of code need serious rethinking as we do not take into account the three-hop, 4-hop, etc.
                    # trajectories to reach a neuron. The probability of this event might be rare but if we plan not to ignore it,
                    # we'd better modify the code. Or, just consider neighboring layers. 
                    file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out-1)
                
                    file_name = file_name_base_data + "/Graphs/De_MLFF_%s.txt" %file_name_ending_temp
                    De = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(De.shape) == 1):
                        De = De.reshape(n_exc,1)
                
                    file_name = file_name_base_data + "/Graphs/Di_MLFF_%s.txt" %file_name_ending_temp
                    Di = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(Di.shape) == 1):
                        Di = Di.reshape(n_inh,1)
                    
                    D_lay_1 = np.vstack((De,Di))
                
                
                    file_name_ending_temp = file_name_ending + '_l_' + str(l_in+1) + '_to_' + str(l_out)
                    
                    file_name = file_name_base_data + "/Graphs/De_MLFF_%s.txt" %file_name_ending_temp
                    De = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(De.shape) == 1):
                        De = De.reshape(n_exc,1)        
                
                    file_name = file_name_base_data + "/Graphs/Di_MLFF_%s.txt" %file_name_ending_temp
                    Di = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(Di.shape) == 1):
                        Di = Di.reshape(n_inh,1)
                
                    D_lay_2 = np.vstack((De,Di))
                
                
                    file_name_ending_temp = file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
                    
                    file_name = file_name_base_data + "/Graphs/De_MLFF_%s.txt" %file_name_ending_temp
                    De = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    if (len(De.shape) == 1):
                        De = De.reshape(n_exc,1)
                
                    file_name = file_name_base_data + "/Graphs/Di_MLFF_%s.txt" %file_name_ending_temp
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
                        
                        file_name = file_name_base_data + "/Graphs/De_MLFF_%s.txt" %file_name_ending_temp
                        np.savetxt(file_name,De,'%1.4f',delimiter='\t',newline='\n')
                        file_name = file_name_base_data + "/Graphs/Di_MLFF_%s.txt" %file_name_ending_temp
                        np.savetxt(file_name,Di,'%1.4f',delimiter='\t',newline='\n')
                        #..........................................................
        #--------------------------------------------------------------------------
    
    #-------------------Run the Network and Record Spikes----------------------
    file_name_base = file_name_base_data + "/Spikes/S_times_MLFF_%s" %file_name_ending
    
    for l_in in range(0,no_layers+1):
            file_name = file_name_base + '_l_' + str(l_in) +'.txt'
            S_time_file = open(file_name,'w')
            S_time_file.close()
            
    for cascade_count in range(0,no_cascades):
        
        #--------------------------Generate Activity---------------------------
        Neural_Connections_Out = generate_neural_activity('F',Neural_Connections,running_period,file_name_base,neural_model_eq,frac_input_neurons,cascade_count,no_layers)
        #----------------------------------------------------------------------
        
        #------------------------Check for Any Errors--------------------------
        for l_in in range(0,no_layers):
            for l_out in range(l_in,no_layers):
                ind = str(l_in) + str(l_out)
                
                temp_list1 = Neural_Connections[ind]
                temp_list2 = Neural_Connections_Out[ind]
                
                if norm(temp_list1[0]-temp_list2[0]):
                    print('something is wrong with We!')
                    break
                if norm(temp_list1[2]-temp_list2[2]):
                    print('something is wrong with Wi!')
                    break
                if norm(temp_list1[1]-temp_list2[1]):
                    print('something is wrong with De!')
                    break
                if norm(temp_list1[3]-temp_list2[3]):
                    print('something is wrong with Di!')
                    break
        #----------------------------------------------------------------------
#==============================================================================
