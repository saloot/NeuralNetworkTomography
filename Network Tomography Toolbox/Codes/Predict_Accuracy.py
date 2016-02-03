#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import sys,getopt,os
try:
    import matplotlib.pyplot as plt
except:
    print 'Matplotlib can not be initiated! Pas de probleme!'
import pdb
from copy import deepcopy
from scipy.cluster.vq import whiten
from scipy.signal import find_peaks_cwt
import os.path

from CommonFunctions.auxiliary_functions import read_spikes,combine_weight_matrix,combine_spikes_matrix,generate_file_name,spike_binning
from CommonFunctions.auxiliary_functions_inference import *
from CommonFunctions.Neurons_and_Networks import *

#from sklearn import metrics

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:X:Y:C:V:J:U:Z:b:p:j:o:")

frac_stimulated_neurons,no_stimul_rounds,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,file_name_base_results,inference_method,sparsity_flag,we_know_topology,verify_flag,beta,alpha0,infer_itr_max,p_miss,jitt,bin_size,neuron_range = parse_commands_inf_algo(input_opts)
#==============================================================================


#================================INITIALIZATIONS===============================

#---------------------Initialize Simulation Variables--------------------------
theta = 20#.005                                               # The update threshold of the neurons in the network
d_window = 2                                          # The time window the algorithm considers to account for pre-synaptic spikes
sparse_thr0 = 0.005                                    # The initial sparsity soft-threshold (not relevant in this version)
max_itr_optimization = 250                              # This is the maximum number of iterations performed by internal optimization algorithm for inference
tau_d = 20.0                                    # The decay time coefficient of the neural membrane (in the LIF model)
tau_s = 2.0                                     # The rise time coefficient of the neural membrane (in the LIF model)
#------------------------------------------------------------------------------

#-------------------------Initialize Inference Parameters----------------------
inference_method = 7
sparsity_flag = 5
n_ind = 1                       # Index of the neuron which we want to infer
#------------------------------------------------------------------------------

#--------Calculate the Range to Assess the Effect of Recording Duration--------
T_max = 500000
#T_max = int(1000*T_max)
#T_step = int(T_max/6.0)
#T_range = range(T_step, T_max+1, T_step)

T_range = [5000]

#------------------------------------------------------------------------------

#----------------------Initialize the Results Matrices-------------------------
Prediction_true_pos = np.zeros([len(T_range),1])
Prediction_true_neg = np.zeros([len(T_range),1])
#------------------------------------------------------------------------------

#-------------Specify the Name of the Files Containg the Spikes----------------
file_name_integrated_spikes_base = '../Data/Spikes/Moritz_Integrated_750'
file_name_spikes = '../Data/Spikes/Moritz_Spike_Times.txt'
file_name_prefix = 'Moritz'
file_name_spikes = '../Data/Spikes/HC3_ec013_198_processed.txt'
file_name_prefix = 'HC3'
file_name_spikes2 = file_name_spikes[:-4] + '_file.txt'
#------------------------------------------------------------------------------

#==============================================================================


#=======================VERIFY PREPROCESSING STEP IS DONE====================
if not os.path.isfile(file_name_spikes2):
    print "The preprocessing step is not done. Please run the following command first and then excute the current file"
    exit 
#==============================================================================

#====================READ THE GROUND TRUTH IF POSSIBLE=========================
#file_name = '../Data/Graphs/network_mocktest_adapted.txt'
#file_name = '../Data/Graphs/Matrix_Accurate.txt'
file_name = '../Data/Graphs/Moritz_Actual_Connectivity.txt'
#file_name = '../Data/Graphs/Connectivity_Matrix2.txt'

W_act = np.genfromtxt(file_name, dtype=None, delimiter='\t')
W_act = W_act.T
n,m = W_act.shape
DD_act = 1.5 * abs(np.sign(W_act))
#file_name = '../Data/Graphs/Delay_Matrix2.txt'
#D_act = np.genfromtxt(file_name, dtype=None, delimiter='\t')
#DD_act = np.multiply(np.sign(abs(W_act)),D_act)
#DD_act = (np.ceil(1000 * DD_act)).astype(int)

#==============================================================================

#=======================EVALUATE THE PREDICTION ACCURACY=======================

#--------------------------Infer the Graph For Each T--------------------------
itr_T = 0
for T in T_range:
    
    file_name_ending = 'I_' + str(inference_method) + '_S_' + str(sparsity_flag) + '_T_' + str(T)
    if p_miss:
        file_name_ending = file_name_ending + '_pM_' + str(p_miss)
    if jitt:
        file_name_ending = file_name_ending + '_jt_' + str(jitt)
    if bin_size:
        file_name_ending = file_name_ending + '_bS_' + str(bin_size)
   
    file_name =  file_name_base_results + "/Inferred_Graphs/W_Pll_%s_%s_%s.txt" %(file_name_prefix,file_name_ending,str(n_ind))
    W = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #W = W_act[:,n_ind]
    
    T_test = int(0.2*T)
    T0 = T
    T0 = 2000
    T_array = [[T0+100,T0+100+T_test],[2*T0+100,2*T0+100+T_test],[10*T0+100,10*T0+100+T_test]]
    
    W2 = np.zeros([94,1])
    W2[0:93,0] = W[0:93]
    W = np.array(W2)
    W = W - W.mean()    
    Accur_true_pos,Accur_true_neg = spike_pred_accuracy(file_name_spikes2,T_array,W,n_ind,theta)
    Prediction_true_pos[itr_T,0] = Accur_true_pos
    Prediction_true_neg[itr_T,0] = Accur_true_neg
    itr_T = itr_T + 1    
    
    pdb.set_trace()
    print 'Evaluation is successfully completed for T = %s ms' %str(T/1000.0)
#------------------------------------------------------------------------------

#==============================================================================


#=====================STORE THE RESULTS TO THE FILE============================

file_name_ending = 'I_' + str(inference_method) + '_S_' + str(sparsity_flag) + '_T_' + str(T)
if p_miss:
    file_name_ending = file_name_ending + '_pM_' + str(p_miss)
if jitt:
    file_name_ending = file_name_ending + '_jt_' + str(jitt)
if bin_size:
    file_name_ending = file_name_ending + '_bS_' + str(bin_size)
            
file_name =  file_name_base_results + "/Plot_Results/Pred_Acc_%s_%s.txt" %(file_name_ending,str(n_ind))
temp = np.hstack([Prediction_true_pos,Prediction_true_neg])
T_range = np.reshape(T_range,[len(T_range),1])
T_range = np.divide(T_range,1000).astype(int)
temp = np.hstack([T_range,temp])
np.savetxt(file_name,temp,'%2.5f',delimiter='\t')

pdb.set_trace()
#==============================================================================

#--------------------------Calculate Accuracy-------------------------
#fpr, tpr, thresholds = metrics.roc_curve(WW.ravel(),whiten(W_inferred).ravel())    
#print('\n==> AUC = '+ str(metrics.auc(fpr,tpr))+'\n');
#----------------------------------------------------------------------
    
    


