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


#================================INITIALIZATIONS===============================

#---------------------Initialize Simulation Variables--------------------------
n = 94                                          # Number of neurons in the dataset
eta = 0.5
eta_max = 0.1
#------------------------------------------------------------------------------

#==============================================================================

#================Read the Degree Distribution of the Dataset===================
file_name_W_deg = '../Data/Graphs/HC3_W_deg.txt'
W_deg = np.genfromtxt(file_name_W_deg, dtype=None, delimiter='\t')
    
neurons_type_sum = np.zeros([len(Var1_range),n])
neurons_type_max = np.zeros([len(Var1_range),n])
acc_neurons_type_sum = np.zeros([len(Var1_range),1])
acc_neurons_type_max = np.zeros([len(Var1_range),1])
false_neurons_type_sum = np.zeros([len(Var1_range),1])
false_neurons_type_max = np.zeros([len(Var1_range),1])
neurons_type_actual = np.zeros([n,1])
for ik in range(0,n):
    if W_deg[ik,0] > W_deg[ik,1]:
        neurons_type_actual[ik,0] = 1
    elif W_deg[ik,0] < W_deg[ik,1]:
        neurons_type_actual[ik,0] = -1
#==============================================================================

#============================Read the  Inferred Weights========================
W_inferred = np.zeros([n,n])
file_name_base = "W_Pll_HC3_ec013_198_processed_I_1_S_1.0_C_8_B_300000_K_E_H_0.0_ii_2_***_T_1200000.txt"
for i in range(0,n):
    file_name = "../Results/Inferred_Graphs/" + file_name_base.replace('***',str(i))
    W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    W_inferred[:,i] = W_read[0:n]
#==============================================================================

#====================Transform the Inferred Matrix to Ternary==================

#==============================================================================

#================Estmate Neuron Type from Inferred Weights=====================
neurons_type_inferred = np.zeros([n,1])
aa = sum(W_inferred.T)

true_pos_exc = 0
true_pos_inh = 0
false_pos_exc = 0
false_pos_inh = 0

for i in range(0,n):
    W_r = W_inferred[i,:]
    deg_ind = neurons_type_actual[i]
    neurons_type_inferred[i] = np.sign(aa[i])
    
    if deg_ind > 0:
        if aa[i] > 0:
            true_pos_exc = true_pos_exc + 1
        else:
            false_pos_inh = false_pos_inh + 1
        
    elif deg_ind < 0:
        if aa[i] < 0:
            true_pos_inh = true_pos_inh + 1
        else:
            false_pos_exc = false_pos_exc + 1
#==============================================================================

#===========================Calculate Accuracy=================================
no_exc = sum(neurons_type_actual>0)
no_inh = sum(neurons_type_actual<0)

recal_exc = true_pos_exc/float(no_exc)
recal_inh = true_pos_inh/float(no_inh)

precision_exc = true_pos_exc/float(sum(aa>0))
precision_inh = true_pos_inh/float(sum(aa<0))

#==============================================================================
else:
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
                
    



    

