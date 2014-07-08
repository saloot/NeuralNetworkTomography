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

no_samples_per_cascade = 3.0              # Number of samples that will be recorded
running_period = (no_samples_per_cascade/10.0)  # Total running time in seconds

no_cascades = 1500                    # Number of times we inject stimulus to the network
ensemble_size = 1                       # The number of random networks that will be generated

theta = 10

Delta = 0.1

#----------------------------------Neural Model--------------------------------
tau=10*ms
tau_e=2*ms # AMPA synapse
eqs='''
dv/dt=(I-v)/tau : volt
dI/dt=-I/tau_e : volt
'''
#------------------------------------------------------------------------------


def myrates(t):
    rates=zeros(n)*Hz    
    if t < 0.1 * ms:
        input_index = floor(n*rand(round(n*frac_input_neurons)))
        input_index = input_index.astype(int)
        rates[input_index]=ones(round(n*frac_input_neurons))*input_stimulus_freq *Hz
    return rates

def q_func(x):
    x = x.astype(float)
    return 0.5 - 0.5 * erf(x/sqrt(2))
        
    
#==============================================================================




for ensemble_count in range(0,ensemble_size):

#======================READ THE NETWORK AND SPIKE TIMINGS======================
    
    #----------------------Construct Prpoper File Names------------------------
    file_name_ending = "n_exc_%s" %str(int(n_exc))
    file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
    file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
    file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
    file_name_ending = file_name_ending + "_d_%s" %str(synapse_delay)
    file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
    file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
    #--------------------------------------------------------------------------
    
    
    #------------------------Read the Input Matrix-----------------------------
    file_name = "/Hesam/Academic/Network Tomography/Data/Graphs/We_cascades_%s.txt" %file_name_ending
    We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    file_name = "/Hesam/Academic/Network Tomography/Data/Graphs/Wi_cascades_%s.txt" %file_name_ending
    Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
    W = np.vstack((We,Wi))
    #--------------------------------------------------------------------------
        

    #------------------------------Read and Sort Spikes------------------------
    file_name = "/Hesam/Academic/Network Tomography/Data/Spikes/S_times_cascades_%s.txt" %file_name_ending
    
    S_time_file = open(file_name,'r')
    S_times = np.fromfile(file_name, dtype=float, sep='\t')
    
    
    neuron_count = 0
    cascade_count = 0
    in_spikes = np.empty([n,no_cascades])
    out_spikes = np.empty([n,no_cascades])
    for l in range(0,len(S_times)):
        if (S_times[l] == -1):
            neuron_count = neuron_count + 1
        elif (S_times[l] == -2):
            neuron_count = 0
            cascade_count = cascade_count + 1
        else:
            if (S_times[l] < 0.00015):
                in_spikes[neuron_count][cascade_count] = 1
            else:
                out_spikes[neuron_count][cascade_count] = 1
    #--------------------------------------------------------------------------            
            
    #---------------------Check for Conflicts in Data--------------------------
    if (sum(multiply(in_spikes,out_spikes)) > 0):
        print('Error! Do something!')
        sys.exit()
    #--------------------------------------------------------------------------
        
#==============================================================================

#for i in range(0,10):
#    R2 = -pow(-1,out_spikes_temp[i,:])
#    print sum(np.divide(np.multiply(R2,S),L_temp))

#============================INFER THE CONNECTIONS=============================
    out_spikes_neg = -pow(-1,out_spikes)
    L = in_spikes.sum(axis=0)
    L_out = out_spikes.sum(axis=0)
    for T in range(100, no_cascades, 400):
        L_temp = L[0:T]
        mu = (p_plus - p_minus) * L_temp
        var = sqrt((p_plus * (1 - p_plus) + p_minus * (1-p_minus) + 2 * p_plus * p_minus) * L_temp)
        out_spikes_neg_temp = out_spikes_neg[:,0:T]
        out_spikes_temp = out_spikes[:,0:T]
        in_spikes_temp = in_spikes[:,0:T]
        W_inferred = np.empty([n,n])
        for i in range(0,n):
            R = out_spikes_temp[i,:]
            for j in range(0,n):
                S = in_spikes_temp[j,:]
                
                mu_p = (p_plus - p_minus) * (L_temp - S)
                var_p = sqrt((p_plus * (1 - p_plus) + p_minus * (1-p_minus) + 2 * p_plus * p_minus) * (L_temp - S) )
                
                denom = q_func((theta - mu)/var)
                nom_plus = q_func((theta - mu_p - S)/var_p)
                nom_minus = q_func((theta - mu_p + S)/var_p)
                nom_zero = q_func((theta - mu)/var)
                
                p_W_plus_1 = p_plus * np.divide(nom_plus, denom)
                p_W_zero_1 = (1-p_plus-p_minus) * np.divide(nom_zero, denom)
                p_W_minus_1 = p_minus * np.divide(nom_minus, denom)
                
                p_W_plus_0 = p_plus * np.divide(1-nom_plus, 1-denom)
                p_W_zero_0 = (1-p_plus-p_minus) * np.divide(1-nom_zero, 1-denom)
                p_W_minus_0 = p_minus * np.divide(1-nom_minus, 1-denom)
                
                p_W_plus = np.multiply(R,p_W_plus_1) + np.multiply(1-R,p_W_plus_0)                
                p_W_zero = np.multiply(R,p_W_zero_1) + np.multiply(1-R,p_W_zero_0)
                p_W_minus = np.multiply(R,p_W_minus_1) + np.multiply(1-R,p_W_minus_0)
                
                p_W_plus = p_W_plus/p_plus
                p_W_minus = p_W_minus/p_minus
                p_W_zero = p_W_zero/(1-p_plus-p_minus)
                LLR_W_plus = log(np.divide(p_W_plus , p_W_minus + p_W_zero))
                LLR_W_minus = log(np.divide(p_W_minus , p_W_plus + p_W_zero))
                LLR_W_zero = log(np.divide(p_W_zero , p_W_plus + p_W_minus))
                
                B_plus = sum(np.multiply(p_W_plus>p_W_zero,p_W_plus>p_W_minus))                 # The number of times we believed a weight is +1
                B_minus = sum(np.multiply(p_W_minus>p_W_zero,p_W_plus<p_W_minus))               # The number of times we believed a weight is -1
                B_zero = sum(np.multiply(p_W_minus<p_W_zero,p_W_plus<p_W_zero))                 # The number of times we believed a weight is 0
                #B_0 = T - B_plus - B_minus
                W_inferred[j,i] = sum(B_plus*Delta - B_minus * Delta)
                
            W_inferred[i,i] = 0
#==============================================================================


#=============================TRANSFORM TO BINARY==============================
        W_binary = np.empty([n,n])
        for i in range(0,n):
            w_temp = W_inferred[:,i]
            ind = np.argsort(w_temp)
            w_temp = zeros(n)
            w_temp[ind[0:int(round(p_minus*n))]] = -1
            w_temp[ind[len(ind)-int(round(p_plus*n))+1:len(ind)]] = 1
            W_binary[:,i] = w_temp
#==============================================================================
        

#=============================CALCULATE ACCURACY===============================
        acc_plus = float(sum(multiply(W_binary>0,W>0)))/float(sum(W>0))
        acc_minus = float(sum(multiply(W_binary<0,W<0)))/float(sum(W<0))
        acc_zero = float(sum(multiply(W_binary==0,W==0)))/float(sum(W==0))
        file_name_ending_new = file_name_ending + "_%s" %str(T)
        file_name = "/Hesam/Academic/Network Tomography/Results/Accuracies/Acc_%s.txt" %file_name_ending_new
        acc_file = open(file_name,'a')
        acc_file.write("%f \t" % acc_plus)
        acc_file.write("%f \t" % acc_minus)
        acc_file.write("%f \t" % acc_zero)
        acc_file.write("\n")
        acc_file.close()

#==============================================================================
    