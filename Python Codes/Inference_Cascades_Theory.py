#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import time
import numpy as np
import sys
from scipy.special import erf
#==============================================================================


#================================INITIALIZATIONS===============================
n_exc = 320                             # The number of excitatory neurons in the output layer
n_inh = 80                              # The number of inhibitory neurons in the output layer
n = n_exc + n_inh                       # Total number of neurons in the output layer

synapse_delay = 0.0                     # The synaptic delay of ALL links in ms

connection_prob = 0.15                  # The probability of having a link from the input to output neurons in the second layer
p_minus = connection_prob * (float(n_inh)/float(n))
p_plus = connection_prob * (float(n_exc)/float(n))

input_stimulus_freq = 20000               # The frequency of spikes by the neurons in the input layer (in Hz)

frac_input_neurons = .25                # Fraction of neurons in the input layer that will be excited by a stimulus

no_samples_per_cascade = 3.0              # Number of samples that will be recorded
running_period = (no_samples_per_cascade/10.0)  # Total running time in seconds

no_cascades = 10000                    # Number of times we inject stimulus to the network
ensemble_size = 1                       # The number of random networks that will be generated

theta = 10  

binary_mode = 2
epsilon = 0.45
file_name_base_data = "./Data"       #The folder to read neural data from
file_name_base_resuts = "./Results"
#------------------------Construct Prpoper File Names--------------------------
file_name_ending = "n_exc_%s" %str(int(n_exc))
file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
file_name_ending = file_name_ending + "_d_%s" %str(synapse_delay)
file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
#------------------------------------------------------------------------------

def q_func(x):
    x = x.astype(float)
    return 0.5 - 0.5 * erf(x/sqrt(2))
#==============================================================================        
    

#============================COMPUTE THEORETICAL ACCURACIES=============================



#------------------Calculate the Overall Probability of Correct Decisions----------------
range_T = range(100, no_cascades, 400)

temp = range(1,99)
q_range = [float(x)/float(100) for x in temp]
range_T = q_range

P_C_plus_thr = zeros(len(range_T))
P_C_minus_thr = zeros(len(range_T))
P_C_zero_thr = zeros(len(range_T))
Q1_itr = zeros(len(range_T))
Q2_itr = zeros(len(range_T))
Q3_itr = zeros(len(range_T))
    
#--------------------Calculate the Probabilities of Belief Updates--------------------
#q = sum(in_spikes)/no_cascades/n
q = frac_input_neurons
itr = 0

for q in q_range:
    T = 8000
    no_input_ones = q * n

    mu = (p_plus - p_minus) * no_input_ones
    var = sqrt((p_plus * (1 - p_plus) + p_minus * (1-p_minus) + 2 * p_plus * p_minus) * no_input_ones)
    mu_p = (p_plus - p_minus) * (no_input_ones - 1)
    var_p = sqrt((p_plus * (1 - p_plus) + p_minus * (1-p_minus) + 2 * p_plus * p_minus) * (no_input_ones - 1) )
    
    p_W_plus_1 = q_func((theta - mu_p - 1)/var_p)
    p_W_zero_1 = q_func((theta - mu_p)/var_p)
    p_W_minus_1 = q_func((theta - mu_p + 1)/var_p)
                        
                    
    p_W_plus_0 = 1 - p_W_plus_1
    p_W_zero_0 = 1 - p_W_zero_1
    p_W_minus_0 = 1 - p_W_minus_1
                                        
    e1= p_W_plus_1 - p_W_plus_0
    e2 = p_W_minus_1 - p_W_minus_0
    epsilon_plus_base = e1 - (e1-e2)/4
    epsilon_minus_base = e2 + (e1-e2)/4
    # p_W_plus = p_W_plus/p_plus
    # p_W_minus = p_W_minus/p_minus
    # p_W_zero = p_W_zero/(1-p_plus-p_minus)
    #----------------------------------------------------------------------------------------

    #------------------Calculate the Probability of Correct Decisions per Cascade------------
    P_C_plus_casc = q * p_W_plus_1
    P_E_plus_casc = q * p_W_plus_0
    P_C_minus_casc = q * p_W_minus_0
    P_E_minus_casc = q * p_W_minus_1
    P_C_zero_casc = q * p_W_zero_1
    P_E_zero_casc = q * p_W_zero_0
    #----------------------------------------------------------------------------------------

    c1 = q*(1-q)*(p_W_minus_0 + p_W_plus_1)
    c2 = q*(1-q)*(p_W_minus_1 + p_W_plus_0)

    cc1 = q*(1-q)*(p_W_zero_0 + p_W_plus_1)    
    cc2 = q*(1-q)*(p_W_zero_1 + p_W_plus_0)
    
    ccc1 = q*(1-q)*(p_W_minus_0 + p_W_zero_1)
    ccc2 = q*(1-q)*(p_W_minus_1 + p_W_zero_0)
    
    #----------------------------------------------------------------------------------------
    

    
    Delta = 1.0/T
        
    if (binary_mode == 1):
        epsilon_plus = epsilon_plus_base * Delta * T * q
        epsilon_minus = epsilon_minus_base * Delta * T * q
    
        mu_plus = T * (P_C_plus_casc - P_E_plus_casc) * Delta
        var_plus = sqrt((P_C_plus_casc * (1 - P_C_plus_casc) + P_E_plus_casc * (1-P_E_plus_casc) + 2 * P_C_plus_casc * P_E_plus_casc) * T * Delta * Delta )
    
        mu_minus = T * (P_C_minus_casc - P_E_minus_casc) * (-Delta)
        var_minus = sqrt((P_C_minus_casc * (1 - P_C_minus_casc) + P_E_minus_casc * (1-P_E_minus_casc) + 2 * P_C_minus_casc * P_E_minus_casc) * T * Delta * Delta)
    
        mu_zero = T * (P_C_zero_casc - P_E_zero_casc) * Delta
        var_zero = sqrt((P_C_zero_casc * (1 - P_C_zero_casc) + P_E_zero_casc * (1-P_E_zero_casc) + 2 * P_C_zero_casc * P_E_zero_casc) * T * Delta * Delta )
    
        epsilon_plus = e_plus_t[itr]
        epsilon_minus = e_minus_t[itr]
        P_C_plus_thr[itr] = P_C_plus_thr[itr] + q_func( (epsilon_plus - mu_plus)/var_plus )
        P_C_minus_thr[itr] = P_C_minus_thr[itr] + 1 - q_func( (epsilon_minus - mu_minus)/var_minus )
        P_C_zero_thr[itr] = P_C_zero_thr[itr] + q_func( (epsilon_minus - mu_zero)/var_zero ) - q_func( (epsilon_plus - mu_zero)/var_zero )
        file_name_ending = file_name_ending + "_e_%s" %str(epsilon)
    elif (binary_mode == 2):
        Q1 = q_func( sqrt(T) * (c1-c2)/sqrt(c1+c2-pow(c1-c2,2)) )
        Q2 = q_func( sqrt(T) * (cc1-cc2)/sqrt(cc1+cc2-pow(cc1-cc2,2)) )
        Q3 = q_func( sqrt(T) * (ccc1-ccc2)/sqrt(ccc1+ccc2-pow(ccc1-ccc2,2)) )
        
        Q1_itr[itr] = Q1
        Q2_itr[itr] = Q2
        Q3_itr[itr] = Q3
        P_C_plus_thr[itr] = P_C_plus_thr[itr] + pow(1 - (1-connection_prob)*Q2 - p_minus * Q1,n-round(p_plus*n)) 
        P_C_minus_thr[itr] = P_C_minus_thr[itr] + pow(1 - (1-connection_prob)*Q3 - p_plus * Q1,n-round(p_minus*n))
        P_C_zero_thr[itr] = P_C_zero_thr[itr] + pow(1 - p_minus*Q3 - p_plus * Q2,n-round(connection_prob*n))
        
    itr = itr + 1

    #----------------------------------------------------------------------------------------

plot(range_T,1-Q1_itr,'b')
plot(range_T,1-Q2_itr,'r')
plot(range_T,1-Q3_itr,'k')
#========================WRITING THE RESULTS TO THE FILE=========================
itr = 0
for T in range_T: 
    file_name_ending_new = file_name_ending + "_%s" %str(T)
    file_name_ending_new = file_name_ending_new + "_%s" %str(binary_mode)
    file_name = file_name_base_resuts + "/Accuracies/Acc_thr_%s.txt" %file_name_ending_new
    acc_file = open(file_name,'a')
    acc_file.write("%f \t" % P_C_plus_thr[itr])
    acc_file.write("%f \t" % P_C_minus_thr[itr])
    acc_file.write("%f \t" % P_C_zero_thr[itr])
    acc_file.write("\n")
    acc_file.close()
    itr = itr + 1
    #----------------------------------------------------------------------------------------

#==============================================================================

    