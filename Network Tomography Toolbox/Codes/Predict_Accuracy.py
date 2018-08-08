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
T_range = [325000]
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
        cmd = 'scp salavati@deneb1.epfl.ch:"~/NeuralNetworkTomography/Network\ Tomography\ Toolbox/Results/Inferred_Graphs/W_Pll_%s_%s_%s.txt" ../Results/Inferred_Graphs/' %(file_name_prefix,file_name_ending,str(n_ind))
        #cmd = 'scp salavati@iclcavsrv2.epfl.ch:"~/Desktop/NeuralNetworkTomography/Network\ Tomography\ Toolbox/Results/Inferred_Graphs/W_Pll_%s_%s_%s.txt" ../Results/Inferred_Graphs/' %(file_name_prefix,file_name_ending,str(ik))                
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
    
    
if file_name_prefix == 'HC3':
    file_name_W_deg = '../../Code to Parse Other Datasets/HC-3/W_deg.txt'
    W_deg = np.genfromtxt(file_name_W_deg, dtype=None, delimiter='\t')
    eta = 0.5
    eta_max = 0.1
    
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
                
                
                #plt.plot(neurons_type_max[itr_V1,:]);plt.plot(neurons_type_sum[itr_V1,:],'r');plt.plot(neurons_type_actual.ravel(),'g');plt.show()
                    
            else:
    


W_r = W_inferred
n_z = np.nonzero(sum(W_r))
W_r2 = W_r[:,n_z[0]]
neurons_type_actual_2 = neurons_type_actual[n_z[0]].ravel()

aa = sum(W_r2)

itr_n = 0
true_pos_exc = 0
true_pos_ing = 0
false_pos_exc = 0
false_pos_ing = 0
for deg_ind in neurons_type_actual_2:
    if deg_ind > 0:
        if aa[itr_n] > 0:
            true_pos_exc = true_pos_exc + 1
        else:
            false_pos_inh = false_pos_inh + 1
        
    elif deg_ind < 0:
        if aa[itr_n] < 0:
            true_pos_inh = true_pos_inh + 1
        else:
            false_pos_exc = false_pos_exc + 1
            
    
    itr_n = itr_n + 1

no_exc = sum(neurons_type_actual_2>0)
no_inh = sum(neurons_type_actual_2<0)


recal_exc = true_pos_exc/no_exc
recal_inh = true_pos_inh/no_inh

precision_exc = true_pos_exc/sum(aa>0)
recal_inh = true_pos_inh/sum(aa<0)

    

