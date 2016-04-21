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
sparsity_flag = 5
get_from_host = 0
max_itr_optimization = 8
#------------------------------------------------------------------------------

#--------Calculate the Range to Assess the Effect of Recording Duration--------
T_max = 500000
#T_max = int(1000*T_max)
#T_step = int(T_max/6.0)
#T_range = range(T_step, T_max+1, T_step)

T_range = [75000,175000,250000,350000,450000,550000]
T_range = [225000,325000,425000]
T_range = [425000,725000,1025000,1325000,1625000,1925000,2225000,2525000,2825000,3125000,3425000,4025000,4325000]
T_range = [1925000,2225000,2525000,2825000,3125000,3425000,4025000,4325000]
T_range = [2225000,2825000,3125000,3425000,3725000,4025000,4625000,5225000]
T_range = [425000,725000,1025000,1325000,1625000,1925000,2225000,2525000,2825000,3125000,3425000,3725000,4025000]
T_range = [325000,625000,925000,1225000,1525000,1825000,2125000,2425000,2725000,3025000]
T_range = range(325000,1225001,300000)
T_range = [625000]
#------------------------------------------------------------------------------

#----------------------Initialize the Results Matrices-------------------------
Prediction_true_pos = np.zeros([len(T_range),1])
Prediction_true_neg = np.zeros([len(T_range),1])
Mean_belief = np.zeros([len(T_range),7])
Pres_Reca = np.zeros([len(T_range),7])
#------------------------------------------------------------------------------

#-------------Specify the Name of the Files Containg the Spikes----------------
file_name_integrated_spikes_base = '../Data/Spikes/Moritz_Integrated_750'
file_name_spikes = '../Data/Spikes/Moritz_Spike_Times.txt'
file_name_prefix = 'Moritz'
#file_name_spikes = '../Data/Spikes/HC3_ec013_198_processed.txt'
#file_name_prefix = 'HC3'
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
#file_name = '../Data/Graphs/Connectivity_Matrix2.txt'

if (inference_method == 7):
    file_name_spikes = '../Data/Spikes/Moritz_Spike_Times.txt'
    file_name = '../Data/Graphs/Moritz_Actual_Connectivity.txt'
    file_name_prefix = 'Moritz'
    n_ind = 0                       # Index of the neuron which we want to infer
elif (inference_method == 5):
    inference_method = 7
    file_name_spikes = '../Data/Spikes/HC3_ec013_198_processed.txt'
    file_name_prefix = 'HC3'
    n_ind = 1                       # Index of the neuron which we want to infer
    

W_act = np.genfromtxt(file_name, dtype=None, delimiter='\t')
W_act = W_act.T
n,m = W_act.shape
DD_act = 1.5 * abs(np.sign(W_act))
#file_name = '../Data/Graphs/Delay_Matrix2.txt'
#D_act = np.genfromtxt(file_name, dtype=None, delimiter='\t')
#DD_act = np.multiply(np.sign(abs(W_act)),D_act)
#DD_act = (np.ceil(1000 * DD_act)).astype(int)
W_infer = np.zeros([len(T_range),len(W_act[:,n_ind])])
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
   
    if 1:
        file_name_ending = file_name_ending + '_ii_' + str(max_itr_optimization)    
    file_name =  file_name_base_results + "/Inferred_Graphs/W_Pll_%s_%s_%s.txt" %(file_name_prefix,file_name_ending,str(n_ind))
    
    if get_from_host:
        cmd = 'scp salavati@deneb1.epfl.ch:"~/NeuralNetworkTomography/Network\ Tomography\ Toolbox/Results/Inferred_Graphs/W_Pll_%s_%s_%s.txt" ../Results/Inferred_Graphs/' %(file_name_prefix,file_name_ending,str(ik))                
        os.system(cmd)
        W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
    W_infer[itr_T,:] = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #W_infer[itr_T,:] = W_infer[itr_T,:]/np.linalg.norm(W_infer[itr_T,:])
    #W_infer[itr_T,:] = W_infer[itr_T,:]-W_infer[itr_T,:].mean()
    if itr_T:
        W = sum(W_infer[0:itr_T+1,:])/float(itr_T+1)
        #W = W_infer[itr_T,:]
    else:
        W = W_infer[itr_T,:]

    
    #W = W/np.linalg.norm(W)
    theta = W[-1]
    
    #W = W_act[:,n_ind]
    
    T_test = int(0.2*T)
    T_test = 10000
    T0 = T + 10000
    #T0 = 50
    
    T_array = [[T0+100,T0+100+T_test],[2*T0+100,2*T0+100+T_test],[10*T0+100,10*T0+100+T_test]]
    T_array = [[T0,T0+T_test]]
    
    if file_name_prefix == 'HC3':
        W2 = np.zeros([94,1])
    elif file_name_prefix == 'Moritz':
        W2 = np.zeros([1000,1])
        W2[0:999,0] = W[0:999]
        
    W = np.array(W2)
    
    #W = W - W.mean()
    
    #-------------------------Evaluate Accuracy of Weights-------------------------
    ww = W_act[:,n_ind]
    W = W - W.mean()
    W = W/np.linalg.norm(W)
    Mean_belief[itr_T,0] = (T/1000.)
    Mean_belief[itr_T,1] = sum(np.multiply(ww>0,W.ravel()))/sum(ww>0)
    Mean_belief[itr_T,2] = sum(np.multiply(ww==0,W.ravel()))/sum(ww==0)
    Mean_belief[itr_T,3] = sum(np.multiply(ww<0,W.ravel()))/sum(ww<0)
    
    Mean_belief[itr_T,4] = pow(sum(pow(np.multiply(ww>0,W.ravel())-Mean_belief[itr_T,1],2))/sum(ww>0),0.5)
    Mean_belief[itr_T,5] = pow(sum(pow(np.multiply(ww==0,W.ravel())-Mean_belief[itr_T,2],2))/sum(ww==0),0.5)
    Mean_belief[itr_T,6] = pow(sum(pow(np.multiply(ww<0,W.ravel())-Mean_belief[itr_T,3],2))/sum(ww<0),0.5)
    
    
    W_b = (W>0.99*Mean_belief[itr_T,1]).astype(int) - (W<0.99*Mean_belief[itr_T,3]).astype(int)
    W_b = W_b.ravel()
    Pres_Reca[itr_T,0] = (T/1000.)
    
    Pres_Reca[itr_T,1] = sum(np.multiply((W_b>0).astype(int),(W_act[:,n_ind]>0).astype(int)))/float(sum((W_act[:,n_ind]>0).astype(int)))
    Pres_Reca[itr_T,2] = sum(np.multiply((W_b==0).astype(int),(W_act[:,n_ind]==0).astype(int)))/float(sum((W_act[:,n_ind]==0).astype(int)))
    Pres_Reca[itr_T,3] = sum(np.multiply((W_b<0).astype(int),(W_act[:,n_ind]<0).astype(int)))/float(sum((W_act[:,n_ind]<0).astype(int)))
    
    Pres_Reca[itr_T,4] = sum(np.multiply((W_b>0).astype(int),(W_act[:,n_ind]>0).astype(int)))/float(sum(W_b>0))
    Pres_Reca[itr_T,5] = sum(np.multiply((W_b==0).astype(int),(W_act[:,n_ind]==0).astype(int)))/float(sum(W_b==0))
    Pres_Reca[itr_T,6] = sum(np.multiply((W_b<0).astype(int),(W_act[:,n_ind]<0).astype(int)))/float(sum(W_b<0))
    #------------------------------------------------------------------------------
    
    #W = W_act[:,n_ind]
    Accur_true_pos,Accur_true_neg = spike_pred_accuracy(file_name_spikes2,T_array,W,n_ind,theta)
    Prediction_true_pos[itr_T,0] = Accur_true_pos
    Prediction_true_neg[itr_T,0] = Accur_true_neg
    
    
    print 'Evaluation is successfully completed for T = %s ms' %str(T/1000.0)
    print 'Accuracy were: %s,%s' %(str(Prediction_true_pos[itr_T,0]),str(Prediction_true_neg[itr_T,0]))
    print 'Mean beliefs were'
    print Mean_belief[itr_T,1:]
    print '-----------------------'
    
    itr_T = itr_T + 1    
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


file_name =  file_name_base_results + "/Plot_Results/Mean_Beliefs_%s_%s.txt" %(file_name_ending,str(n_ind))
np.savetxt(file_name,Mean_belief,'%3.5f',delimiter='\t')


file_name =  file_name_base_results + "/Plot_Results/Preci_Reca_%s_%s.txt" %(file_name_ending,str(n_ind))
np.savetxt(file_name,Pres_Reca,'%3.5f',delimiter='\t')
#==============================================================================

pdb.set_trace()
plt.plot(Mean_belief[:,0],Mean_belief[:,1],'r');plt.plot(Mean_belief[:,0],Mean_belief[:,2],'g');plt.plot(Mean_belief[:,0],Mean_belief[:,3],'b');plt.show()
#plt.plot(Mean_belief[:,0],Mean_belief[:,4],'r');plt.plot(Mean_belief[:,0],Mean_belief[:,5],'g');plt.plot(Mean_belief[:,0],Mean_belief[:,6],'b');plt.show()
plt.plot(Pres_Reca[:,0],Pres_Reca[:,1],'r');plt.plot(Pres_Reca[:,0],Pres_Reca[:,2],'g');plt.plot(Pres_Reca[:,0],Pres_Reca[:,3],'b');plt.show()
plt.plot(Pres_Reca[:,0],Pres_Reca[:,4],'r');plt.plot(Pres_Reca[:,0],Pres_Reca[:,5],'g');plt.plot(Pres_Reca[:,0],Pres_Reca[:,6],'b');plt.show()
#W_b = (W>0.07).astype(int) - (W<-0.2).astype(int)
#--------------------------Calculate Accuracy-------------------------
#fpr, tpr, thresholds = metrics.roc_curve(WW.ravel(),whiten(W_inferred).ravel())    
#print('\n==> AUC = '+ str(metrics.auc(fpr,tpr))+'\n');
#----------------------------------------------------------------------
    
    



