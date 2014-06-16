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

frac_input_neurons = .23                # Fraction of neurons in the input layer that will be excited by a stimulus


no_cascades = 10000                    # Number of times we inject stimulus to the network
ensemble_size = 1                       # The number of random networks that will be generated

theta = 10  

binary_mode = 2

def q_func(x):
    x = x.astype(float)
    return 0.5 - 0.5 * erf(x/sqrt(2))
#==============================================================================        
    

#============================COMPUTE THEORETICAL ACCURACIES=============================



#------------------Calculate the Overall Probability of Correct Decisions----------------
temp = range(1,99)
q_range = [float(x)/float(100) for x in temp]

n_range = range(50,1000,50)
Delta_Q = zeros([len(n_range),len(q_range)])
Delta_Q_mn = zeros([len(n_range),len(q_range)])
q_opt = zeros(len(n_range))
val_opt = zeros(len(n_range))

itr_n = 0
for n in n_range:
    itr = 0
    for q in q_range:    
        mu = q*n*(p_plus-p_minus)
        sigma = sqrt(q*n*(p_plus+p_minus)-pow(p_plus-p_minus,2))
        Delta_Q[itr_n,itr] = q_func((theta-mu-1)/sigma)-q_func((theta-mu)/sigma)
        Delta_Q_mn[itr_n,itr] = q_func((theta-mu)/sigma)-q_func((theta-mu+1)/sigma)
        itr = itr + 1
    
    Delta_Q_tot = Delta_Q + Delta_Q_mn
    
    temp = np.sort(Delta_Q_tot[itr_n,:])
    ind = np.argsort(Delta_Q_tot[itr_n,:])
    
    q_opt[itr_n] = q_range[ind[len(ind)-1]]
    val_opt[itr_n] = temp[len(temp)-1]
    itr_n = itr_n + 1
    

plot(q_range,Delta_Q[0,:],'b')
plot(q_range,Delta_Q_mn[0,:],'b--')

#plot(q_range,Delta_Q[1,:],'r')
#plot(q_range,Delta_Q_mn[1,:],'r--')

q_thr = np.divide(theta/(p_plus-p_minus),n_range)
plot(n_range,q_opt)
plot(n_range,np.divide(theta/(p_plus-p_minus),n_range),'r')

#==============================================================================


#========================WRITING THE RESULTS TO THE FILE=========================
itr = 0
file_name = "./Results/Optimum_q/q_Optimum_p_%s" %str(connection_prob)
file_name = file_name + "_%s" %str(theta)

file_name_thr = file_name# + "_theory"

file_name = file_name + "_exc.txt"
file_name_thr = file_name_thr + "_inh.txt"

q_file = open(file_name,'w')
q_file_thr = open(file_name_thr,'w')
for q in q_range: 
    q_file.write("%f \t" % q_range[itr])
    q_file.write("%f \t" % Delta_Q[0,itr])    
    q_file.write("\n")
    
    q_file_thr.write("%f \t" % q_range[itr])
    q_file_thr.write("%f \t" % Delta_Q_mn[0,itr])    
    q_file_thr.write("\n")
    
    itr = itr + 1
    #----------------------------------------------------------------------------------------
q_file.close()
q_file_thr.close()
#==============================================================================
    