#=======================IMPORT THE NECESSARY LIBRARIES=========================
from time import time
import numpy as np
import sys,getopt,os
try:
    import matplotlib.pyplot as plt
except:
    print('Matplotlib can not be initiated! Pas de probleme!')
import pdb
from copy import deepcopy
from scipy.cluster.vq import whiten
#from scipy.signal import find_peaks_cwt
import os.path

from CommonFunctions.auxiliary_functions import read_spikes,combine_weight_matrix,combine_spikes_matrix,generate_file_name,spike_binning
from CommonFunctions.auxiliary_functions_inference import *
from CommonFunctions.Neurons_and_Networks import *

#from sklearn import metrics

os.system('clear')                                              # Clear the commandline window
#==============================================================================


#================================INITIALIZATIONS===============================

#---------------------Initialize Simulation Variables--------------------------
n = 94                                          # Number of neurons in the dataset
no_structural_info = 0
eta = 0.5
eta_max = 0.1
ternary_flag = 1                                # If 1, the algorithm bases its analysis on the ternary matrix.
                                                # If 0, it will uses the analog inferred graphs as the basis of analysis.
#------------------------------------------------------------------------------

#==============================================================================

#=========================Apply Structural Information=========================
file_name_ground_truth =  "../Data/Graphs/HC3_Structural_Info.txt"
W_gt = np.genfromtxt(file_name_ground_truth, dtype=None, delimiter='\t')
W_gt = W_gt.T
#W_inferred = np.multiply(W_inferred,W_gt)
#==============================================================================

#============================Read the  Inferred Weights========================
W_inferred = np.zeros([n,n])
#file_name_base = "W_Pll_HC3_ec013_198_processed_I_1_S_1.0_C_8_B_300000_K_E_H_0.0_ii_2_***_T_1200000.txt"
# TODO: make these into a pipline too
if no_structural_info:
    file_name_base = "W_Pll_HC3_ec013_198_processed_I_1_S_8.0_C_4_B_150000_K_E_H_0.5_ii_5_***_f_" + str(no_structural_info) + "_T_1500000"
else:
    file_name_base = "W_Pll_HC3_ec013_198_processed_I_1_S_1.0_C_8_B_300000_K_E_H_0.0_ii_4_***_T_1200000"
    
# TODO: make this based on command line as well
ternary_adjustment_ending = "_1.75_2.0_B_4"

if ternary_flag:
    file_name_base = "W_Binary_" + file_name_base
    if not no_structural_info:
        file_name_base += ternary_adjustment_ending + ".txt"
elif not no_structural_info:
    file_name_base += ".txt"

for i in range(0,n):
    file_name = "../Results/Inferred_Graphs/" + file_name_base.replace('***',str(i))
    
    #---------------Get All the Files with Similar File Name Endings---------------
    tmp_str = file_name_base.replace('***',str(i))
    tmp_str = tmp_str.replace('.txt','')
    file_name_ending_list = []

    results_dir = '../Results/Inferred_Graphs'
    for file in os.listdir(results_dir):
        if file.startswith(tmp_str):
            file_name = os.path.join(results_dir, file)
            if '_n_' not in file_name:
                file_name_ending_list.append(file_name)
    for file_name in file_name_ending_list:
        W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')

    #-----------------Calculate the Binary Matrix From Beliefs-----------------
        if no_structural_info:
            file_name_ending_mod = file_name.replace('W_Binary_','')
            file_name_ending_mod = file_name_ending_mod.replace('../Results/Inferred_Graphs/','')
            file_name_ending_mod = file_name_ending_mod.replace('W_Pll_','')
            file_name_ending_mod = file_name_ending_mod.replace(ternary_adjustment_ending,'')

            file_name = '../Results/Inferred_Graphs/Hidden_or_Structured_Neurons_' + file_name_ending_mod

            try:
                hidden_neurons = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            except:
                continue
            hidden_neurons = np.hstack([hidden_neurons,i])
        else:
            hidden_neurons = [i]
    #--------------------------------------------------------------------------

    #--------------------Remap the --------------------------
        hidden_neurons = hidden_neurons.astype(int)
        itr = 0
        for j in range(0,n):
            if j not in hidden_neurons:
                W_inferred[j,i] = W_read[itr]
                itr += 1
    #--------------------------------------------------------------------------

#==============================================================================

#================Read the Degree Distribution of the Dataset===================
file_name_W_deg = '../Data/Graphs/HC3_W_deg.txt'
W_deg = np.genfromtxt(file_name_W_deg, dtype=None, delimiter='\t')
    
neurons_type_actual = np.zeros([n,1])
for ik in range(0,n):
    if W_deg[ik,0] > W_deg[ik,1]:#W_deg[ik,8] > 0
        neurons_type_actual[ik,0] = 1
    elif W_deg[ik,0] < W_deg[ik,1]: #W_deg[ik,8] < 0
        neurons_type_actual[ik,0] = -1
#==============================================================================

#================Estmate Neuron Type from Inferred Weights=====================
neurons_type_inferred = np.zeros([n,1])
aa = sum(W_inferred)

true_pos_exc = 0
true_pos_inh = 0
false_pos_exc = 0
false_pos_inh = 0
cutoff_thr = 5
for i in range(0,n):
    W_r = W_inferred[i,:]
    deg_ind = neurons_type_actual[i]
    neurons_type_inferred[i] = np.sign(aa[i])
    #ss = aa[i]  
    ss = (sum(W_r>0)-sum(W_r<0))

    if ss/sum(W_r!=0) > 0.95 and ss > cutoff_thr:
        neurons_type_inferred[i] = 1
    elif ss < -cutoff_thr:
        neurons_type_inferred[i] = -1
    
    if deg_ind > 0:
        if neurons_type_inferred[i] == 1:
            true_pos_exc = true_pos_exc + 1
        elif neurons_type_inferred[i] == -1:
            false_pos_inh = false_pos_inh + 1
        
    elif deg_ind < 0:
        if neurons_type_inferred[i] == -1:
            true_pos_inh = true_pos_inh + 1
        elif neurons_type_inferred[i] == 1:
            false_pos_exc = false_pos_exc + 1
#==============================================================================

#===========================Calculate Accuracy=================================
no_exc = sum(neurons_type_actual>0)
no_inh = sum(neurons_type_actual<0)

recal_exc = true_pos_exc/float(no_exc)
recal_inh = true_pos_inh/float(no_inh)

# To calculate precision, limit attention to those neurons we are
# "SURE" what their type is
if 1:
    for i in range(0,len(neurons_type_actual)):
        if neurons_type_actual[i] == 0:
            neurons_type_inferred[i] = 0

precision_exc = true_pos_exc/float(sum(neurons_type_inferred>0))
precision_inh = true_pos_inh/float(sum(neurons_type_inferred<0))

print(recal_exc,recal_inh,precision_exc,precision_inh)
#==============================================================================
if 0:
    neurons_type_sum = np.zeros([len(Var1_range),n])
    neurons_type_max = np.zeros([len(Var1_range),n])
    acc_neurons_type_sum = np.zeros([len(Var1_range),1])
    acc_neurons_type_max = np.zeros([len(Var1_range),1])
    false_neurons_type_sum = np.zeros([len(Var1_range),1])
    false_neurons_type_max = np.zeros([len(Var1_range),1])

    if file_name_prefix == 'HC3':
                W_r = W_inferred[0:n,0:n]
                #W_r = W_inferred[0:n,0:n]-np.dot(np.ones([n,1]),np.reshape(W_inferred[0:n,0:n].mean(axis = 0),[1,n]))
                #W_r = np.divide(W_inferred[0:n,0:n],np.dot(np.ones([n,1]),np.reshape(pow(0.0001+np.sum(np.multiply(W_inferred[0:n,0:n],W_inferred[0:n,0:n]),axis = 0),0.5),[1,n])))
                #W_r = W_r/np.abs(W_r).max()
                
                
                # V is the membrane potential
                # n is the number of peaks to detect
                # t_fire contains the actual spiking times.
                
                
                n_peak = 0          # The number of counted peaks so far
                eta = 0.75
                
                peaks_neur = np.zeros([n,2])
                for ik in range(0,n):
                    peakvalues = []    # The magnitude of the detected peaks
                    peak_inds = []      # Indices of the peaks
                    ww = W_r[:,ik]
                    mu = abs(ww).mean()
                    st = abs(ww).std()
                    #~~~~~~~~Pick the peaks that have the 90% of Mass~~~~~~~
                    iinds = np.argsort(np.abs(ww))[::-1]
                    ivals = np.sort(np.abs(ww))[::-1]
                    sum_val = sum(ivals)
                    temp_sum = 0
                    itr = 0
                    for itr in range(0,len(ivals)):
                        item = ivals[itr]
                        temp_sum = temp_sum + item
                        
                        #if temp_sum/(1e-10+sum_val) > eta:
                        if itr > eta:
                            break
                        else:
                            iji = iinds[itr]
                            
                            if abs(ww[iji]) > mu+2*st:
                                peakvalues.append(ww[iji])
                                if ww[iji]>0:
                                    peaks_neur[ik,0] = peaks_neur[ik,0] + (ww[iji])
                                else:
                                    peaks_neur[ik,1] = peaks_neur[ik,1] + (ww[iji])
                            
                                itr = itr + 1
                            else:
                                break
                        
                    
                peakvalues = np.array(peakvalues)
                #plt.plot(peaks_neur[:,0]+peaks_neur[:,1]);plt.plot(W_deg[:,0]-W_deg[:,1],'r');plt.show()
                #plt.plot(peaks_neur[:,0]);plt.plot(W_deg[:,4],'r');plt.show()
                plt.plot(peaks_neur[:,0]);plt.plot(W_deg[:,4],'r');plt.show()
                plt.plot(peaks_neur[:,1]);plt.plot(W_deg[:,5],'r');plt.show()
                plt.plot(peaks_neur[:,0]+peaks_neur[:,1]);plt.plot(W_deg[:,4]+W_deg[:,5],'r');plt.show()
                neurons_type_sum[itr_V1,:] = np.sign(peaks_neur[:,0]+peaks_neur[:,1])
                neurons_type_max[itr_V1,:] = np.sign(peaks_neur[:,0]+peaks_neur[:,1])
                        
                
                #Check excitatory connections
                acc_neurons_type_sum[itr_V1,0] = sum(np.multiply((np.sign(W_deg[:,4].ravel()) == np.sign(peaks_neur[:,0])).astype(int),(W_deg[:,4].ravel()!=0).astype(int)))
                acc_neurons_type_sum[itr_V1,0] = acc_neurons_type_sum[itr_V1,0]/(0.0001 + sum(W_deg[:,4].ravel()!=0) )
                
                acc_neurons_type_max[itr_V1,0] = sum(np.multiply((np.sign(W_deg[:,5].ravel()) == np.sign(peaks_neur[:,1])).astype(int),(W_deg[:,5].ravel()!=0).astype(int)))
                acc_neurons_type_max[itr_V1,0] = acc_neurons_type_max[itr_V1,0]/(0.0001 + sum(W_deg[:,5].ravel()!=0) )
                
                if 0:
                    n_clust = 11
                    temp_thr,res = kmeans(W_inferred[:-1,n_ind],n_clust,iter=30);temp_thr.sort();W_temp,res2 = vq(W_inferred[:-1,n_ind],temp_thr);    
                    plt.plot((W_temp==n_clust-1).astype(int) - (W_temp==0).astype(int));plt.show()
                    W_deg[n_ind,:]
                
                acc_neurons_type_sum[itr_V1,0] = sum(np.multiply((neurons_type_actual.ravel() == neurons_type_sum[itr_V1,:]).astype(int),(neurons_type_actual.ravel()!=0).astype(int)))
                acc_neurons_type_sum[itr_V1,0] = acc_neurons_type_sum[itr_V1,0]/(0.0001 + sum(neurons_type_actual.ravel()!=0) )
                
                acc_neurons_type_max[itr_V1,0] = sum(np.multiply((neurons_type_actual.ravel() == neurons_type_max[itr_V1,:]).astype(int),(neurons_type_actual.ravel()!=0).astype(int)))
                acc_neurons_type_max[itr_V1,0] = acc_neurons_type_max[itr_V1,0]/(0.0001 + sum(neurons_type_actual.ravel()!=0) )
    
                false_neurons_type_sum[itr_V1,0]  = sum(np.multiply((neurons_type_actual.ravel() != neurons_type_sum[itr_V1,:]).astype(int),(neurons_type_actual.ravel()!=0).astype(int)))
                false_neurons_type_sum[itr_V1,0] = false_neurons_type_sum[itr_V1,0]/(0.0001 + sum(neurons_type_actual.ravel()!=0) )
                
                
                v_sum = (np.sum(W_r,axis = 1)>eta).astype(int) - (np.sum(W_r,axis = 1)<-eta).astype(int)
                v_max = (np.sum(W_r>eta_max,axis = 1)-np.sum(W_r<-eta_max,axis = 1))
                neurons_type_sum[itr_V1,:] = np.sign(v_sum)
                neurons_type_max[itr_V1,:] = np.sign(v_max)
                
                acc_neurons_type_sum[itr_V1,0] = sum(neurons_type_actual.ravel() == neurons_type_sum[itr_V1,:])
                acc_neurons_type_sum[itr_V1,0] = sum(np.multiply((neurons_type_actual.ravel() == neurons_type_sum[itr_V1,:]).astype(int),(neurons_type_actual.ravel()!=0).astype(int)))
                acc_neurons_type_sum[itr_V1,0] = acc_neurons_type_sum[itr_V1,0]/(0.0001 + sum(neurons_type_actual.ravel()!=0) )
                
                false_neurons_type_sum[itr_V1,0]  = sum(np.multiply((neurons_type_actual.ravel() != neurons_type_sum[itr_V1,:]).astype(int),(neurons_type_actual.ravel()!=0).astype(int)))
                false_neurons_type_sum[itr_V1,0] = false_neurons_type_sum[itr_V1,0]/(0.0001 + sum(neurons_type_actual.ravel()!=0) )
                
                acc_neurons_type_max[itr_V1,0] = sum(neurons_type_actual.ravel() == neurons_type_max[itr_V1,:])
                acc_neurons_type_max[itr_V1,0] = sum(np.multiply((neurons_type_actual.ravel() == neurons_type_max[itr_V1,:]).astype(int),(neurons_type_actual.ravel()!=0).astype(int)))
                acc_neurons_type_max[itr_V1,0] = acc_neurons_type_max[itr_V1,0]/(0.0001 + sum(neurons_type_actual.ravel()!=0) )
                false_neurons_type_max[itr_V1,0]  = sum(np.multiply((neurons_type_actual.ravel() != neurons_type_max[itr_V1,:]).astype(int),(neurons_type_actual.ravel()!=0).astype(int)))
                false_neurons_type_max[itr_V1,0] = false_neurons_type_max[itr_V1,0]/(0.0001 + sum(neurons_type_actual.ravel()!=0) )
                
    



    

