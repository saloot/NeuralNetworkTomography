#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
import time
import pdb
import numpy as np
import sys,getopt,os
from scipy.cluster.vq import kmeans,whiten,kmeans2,vq
try:
    import matplotlib.pyplot as plt
except:
    pass
#from CommonFunctions.auxiliary_functions import combine_weight_matrix,generate_file_name
from CommonFunctions.auxiliary_functions_accuracy import caculate_accuracy,beliefs_to_ternary,parse_commands_accuracy_algo
#from CommonFunctions.Neurons_and_Networks import *

os.system('clear')                                              # Clear the commandline window
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:f:R:L:M:B:R:G:K:C:Y:U:Z:o:N:H:j:c:n:")

file_name_ending_list,file_name_base_results,ternary_mode,file_name_ground_truth,n,no_hidden_neurons,no_structural_connections,adj_fact_exc,adj_fact_inh,n_ind = parse_commands_accuracy_algo(input_opts)

if not file_name_ground_truth:
    print 'Sorry you should specify the file that contains the ground truth.'
    sys.exit()
#==============================================================================

#================================INITIALIZATIONS===============================

#-----------------------Set Simulation Variables------------------------
dale_law_flag = 0   # If 1, the ternarification algorithm returns a matrix in which the non-zero entries in a row (i.e. outgoing neural connections) have the same sign
#------------------------------------------------------------------------------

#---------------------Read The Actual Grapgh If Possible-----------------------
#file_name = '../Data/Graphs/Moritz_Actual_Connectivity.txt'
#file_name = '../Results/Inferred_Graphs/W_Pll_Moritz_I_7_S_5_T_75000_0.txt'
if file_name_ground_truth:
    W = np.genfromtxt(file_name_ground_truth, dtype=None, delimiter='\t')
    W = W.T
    n,m = W.shape
    W_ss = W[:,n_ind]
    W_s = np.zeros([n-no_hidden_neurons-no_structural_connections-1,1])
    #W_s = np.zeros([n,1])
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
recal_exc = np.zeros([len(file_name_ending_list)])
std_recal_exc = np.zeros([len(file_name_ending_list)])
prec_exc = np.zeros([len(file_name_ending_list)])
std_prec_exc = np.zeros([len(file_name_ending_list)])
    
recal_void = np.zeros([len(file_name_ending_list)])
std_recal_void = np.zeros([len(file_name_ending_list)])
prec_void = np.zeros([len(file_name_ending_list)])
std_prec_void = np.zeros([len(file_name_ending_list)])

recal_inh = np.zeros([len(file_name_ending_list)])
std_recal_inh = np.zeros([len(file_name_ending_list)])
prec_inh = np.zeros([len(file_name_ending_list)])
std_prec_inh = np.zeros([len(file_name_ending_list)])

no_roi_steps = 100
true_pos_exc_tot = np.zeros([no_roi_steps])
false_pos_exc_tot = np.zeros([no_roi_steps])

true_pos_inh_tot = np.zeros([no_roi_steps])
false_pos_inh_tot = np.zeros([no_roi_steps])

true_pos_void_tot = np.zeros([no_roi_steps])
false_pos_void_tot = np.zeros([no_roi_steps])
#------------------------------------------------------------------------------
    
#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
W_inferred = np.zeros([n-no_hidden_neurons-no_structural_connections-1,1])
W_infer = np.zeros([len(file_name_ending_list),n-no_hidden_neurons-no_structural_connections-1])

#W_inferred = np.zeros([n,1])
#W_infer = np.zeros([len(file_name_ending_list),n])

#------------------------Read the Ternary Weights-------------------------
itr_i = 0
for file_name_ending in file_name_ending_list:

    file_name = "Inferred_Graphs/" + file_name_ending
    file_name = file_name_base_results + '/' + file_name        
    W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    W_infer[itr_i,:] = W_read[0:n-no_hidden_neurons-no_structural_connections-1]

    #if itr_i > 1:
    #    W_inferred[0:min(m-no_hidden_neurons-no_structural_connections-1,len(W_read)),0] = W_infer[0:itr_i,:].mean(axis = 0)
    #else:
    #    W_inferred[0:min(m-no_hidden_neurons-no_structural_connections-1,len(W_read)),0] = W_infer[itr_i-1,:]
    
    #W_infer[itr_i,:] = W_read[0:n]
    if itr_i > 1:
        W_inferred[0:min(m,len(W_read)),0] = W_infer[0:itr_i,:].mean(axis = 0)
    else:
        W_inferred[0:min(m,len(W_read)),0] = W_infer[itr_i-1,:]

    W_inferred_s = W_inferred
    #--------------------------------------------------------------------------

    #-------------------Transfrom the Results to Ternary-----------------------
    file_name = file_name_base_results + "/Inferred_Graphs/" + file_name_ending

    W_binary = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #--------------------------------------------------------------------------

    #-----------------Calculate the Binary Matrix From Beliefs-----------------
    if no_hidden_neurons or no_structural_connections:
        file_name_ending_mod = file_name_ending.replace('W_Binary_','')
        file_name_ending_mod = file_name_ending_mod.replace('W_Pll_','')
        temp_str = "_" + str(adj_fact_exc) +"_" + str(adj_fact_inh) + "_B_" + str(ternary_mode)
        file_name_ending_mod = file_name_ending_mod.replace(temp_str,'')

        file_name_hidden = "Inferred_Graphs/Hidden_or_Structured_Neurons_" + file_name_ending_mod
        file_name = file_name_base_results + '/' + file_name_hidden
        hidden_neurons = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        hidden_neurons = np.hstack([hidden_neurons,n_ind])
        W_h = np.delete(W_ss,hidden_neurons,0)
    else:
        W_h = np.delete(W_ss,np.array([n_ind]),0)
    
    W_s = W_h

    #pdb.set_trace()
    if no_structural_connections:
        # Re-map the connectivity to the actual matrix
        tmp = np.zeros([n-no_hidden_neurons])
        itr_iij = 0
        for iij in range(0,n-no_hidden_neurons):
            if iij not in hidden_neurons and iij != n_ind:
                tmp[iij] = W_binary[itr_iij]
                itr_iij += 1
    
        W_s = W_ss            
        W_binary = tmp
                    
    #---------Calculate and Display Recall & Precision for Our Method----------    
    recal,precision = caculate_accuracy(W_binary[0:len(W_s)],W_s)

    recal_exc[itr_i] = recal[0]
    recal_inh[itr_i] = recal[1]
    recal_void[itr_i] = recal[2]
        
    prec_exc[itr_i] = precision[0]
    prec_inh[itr_i] = precision[1]
    prec_void[itr_i] = precision[2]
    print(recal_exc[itr_i],recal_inh[itr_i],recal_void[itr_i])
    print('\n')
    print(prec_exc[itr_i],prec_inh[itr_i],prec_void[itr_i])
    print('\n')
    print('\n')
#======================================================================================


#==============================CALCULATE THE ROC CURVE=================================
    
#---------------------------For Excitatory Connections-----------------------------
    ww = W_inferred_s
    ww = ww - ww.mean()
    ww = whiten(ww)
        
    min_val = W_inferred_s.min()
    max_val = W_inferred_s.max()
    mid_val = int(0.5*(max_val+min_val))

    
    val_range_exc = range(0,no_roi_steps)
    val_range_exc = np.array(val_range_exc)*max_val/float(no_roi_steps)
    
    val_range_inh = range(no_roi_steps-1,-1,-1)
    val_range_inh = np.array(val_range_inh)*min_val/float(no_roi_steps)

    true_pos_exc = np.zeros([len(val_range_exc)])
    false_pos_exc = np.zeros([len(val_range_exc)])
    true_pos_inh = np.zeros([len(val_range_inh)])
    false_pos_inh = np.zeros([len(val_range_inh)])
    true_pos_void = np.zeros([min(len(val_range_inh),len(val_range_exc))])
    false_pos_void = np.zeros([min(len(val_range_inh),len(val_range_exc))])

    true_pos_void = np.zeros([min(len(val_range_inh),len(val_range_exc))])
    false_pos_void_tot = np.zeros([min(len(val_range_inh),len(val_range_exc))])

    
    
    W_temp = ww#[:-1]  
    itr_V1 = 0  
    for thresh in val_range_exc:
            
        #~~~~~~~~Calcualte True Positive and False Positive Rates~~~~~~~~~
        W_ter = (W_temp>=thresh).astype(int)

        true_pos_exc[itr_V1] = true_pos_exc[itr_V1] + sum(sum(np.multiply((W_ter>0).astype(int),(W_s>0).astype(int))))/float(sum(W_s>0))
        false_pos_exc[itr_V1] = false_pos_exc[itr_V1] + sum(sum(np.multiply((W_ter>0).astype(int),(W_s<=0).astype(int))))/float(sum(W_s<=0))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        itr_V1 += 1
    
    itr_V1 = 0
    for thresh in val_range_inh:
            
        #~~~~~~~~Calcualte True Positive and False Positive Rates~~~~~~~~~
        W_ter = (W_temp<=thresh).astype(int)
            
        true_pos_inh[itr_V1] = true_pos_inh[itr_V1] + sum(sum(np.multiply((W_ter>0).astype(int),(W_s<0).astype(int))))/float(sum(W_s<0))
        false_pos_inh[itr_V1] = false_pos_inh[itr_V1] + sum(sum(np.multiply((W_ter>0).astype(int),(W_s>=0).astype(int))))/float(sum(W_s>=0))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        itr_V1 = itr_V1 + 1
            
    itr_V1 = 0
    for itr_V1 in range(0,min(len(val_range_inh),len(val_range_exc))):
            
        thr1 = val_range_inh[itr_V1]
        thr2 = val_range_exc[itr_V1]
            
        #~~~~~~~~Calcualte True Positive and False Positive Rates~~~~~~~~~
        W_ter = (W_temp>=thr2).astype(int) + (W_temp<=thr1).astype(int)
        W_ter = (W_ter == 0).astype(int)
            
        true_pos_void[itr_V1] = true_pos_void[itr_V1] + sum(sum(np.multiply((W_ter>0).astype(int),(W_s==0).astype(int))))/float(sum(W_s==0))
        false_pos_void[itr_V1] = false_pos_void[itr_V1] + sum(sum(np.multiply((W_ter>0).astype(int),(W_s!=0).astype(int)))/float(sum(W_s!=0)))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        itr_V1 = itr_V1 + 1 
    #----------------------------------------------------------------------------------        
    
    
    #----------------------------------------------------------------------------------
    true_pos_exc_tot += true_pos_exc
    false_pos_exc_tot += false_pos_exc
    true_pos_inh_tot += true_pos_inh
    false_pos_inh_tot += false_pos_inh
    true_pos_void_tot += true_pos_void
    false_pos_void_tot += false_pos_void
    
    itr_i += 1

true_pos_exc_tot = true_pos_exc_tot/itr_i
false_pos_exc_tot = false_pos_exc_tot/itr_i
true_pos_inh_tot = true_pos_inh_tot/itr_i
false_pos_inh_tot = false_pos_inh_tot/itr_i
true_pos_void_tot = true_pos_void_tot/itr_i
false_pos_void_tot = false_pos_void_tot/itr_i
#======================================================================================

#==================================SAVE THE RESULTS====================================
temp_ending = file_name_ending
try:
    ind = temp_ending.index('_ID_')
    aa = temp_ending[ind:ind+10]
    temp_ending = temp_ending.replace(aa,'')
except:
    pass
temp_ending = temp_ending.replace('W_Binary_','')
temp_ending = temp_ending.replace('W_Pll_','')

recal_exc = recal_exc.mean()
recal_inh = recal_inh.mean()
recal_void = recal_void.mean()

prec_exc = prec_exc.mean()
prec_inh = prec_inh.mean()
prec_void = prec_void.mean() 

temp = np.vstack([recal_exc,recal_inh,recal_void])
temp = temp.T
#temp = np.hstack([np.reshape(var_range,[len(var_range),1]),temp])
file_name = file_name_base_results + "/Accuracies/Rec_" + temp_ending
np.savetxt(file_name,temp,'%f',delimiter='\t',newline='\n')
        
file_name = file_name_base_results + "/Accuracies/Prec_" + temp_ending
temp = np.vstack([prec_exc,prec_inh,prec_void])
temp = temp.T
#temp = np.hstack([np.reshape(var_range,[len(var_range),1]),temp])

np.savetxt(file_name,temp,'%f',delimiter='\t',newline='\n')

file_name = file_name_base_results + "/Plot_Results/ROC_exc_" + temp_ending
np.savetxt(file_name,np.vstack([false_pos_exc_tot,true_pos_exc_tot]).T,'%f',delimiter='\t',newline='\n')
    
file_name = file_name_base_results + "/Plot_Results/ROC_inh_" + temp_ending
np.savetxt(file_name,np.vstack([false_pos_inh_tot,true_pos_inh_tot]).T,'%f',delimiter='\t',newline='\n')
    
file_name = file_name_base_results + "/Plot_Results/ROC_void_" + temp_ending
np.savetxt(file_name,np.vstack([false_pos_void_tot,true_pos_void_tot]).T,'%f',delimiter='\t',newline='\n')
#======================================================================================

#=================================PLOT THE ROC CURVES==================================
val_range = range(0,100)
val_range = np.array(val_range)/100.0
#plt.plot(false_pos_exc,true_pos_exc);plt.plot(val_range,val_range,'r');plt.show()
#plt.plot(false_pos_inh,true_pos_inh);plt.show()#plt.plot(val_range,val_range,'r');plt.show()
#plt.plot(false_pos_void,true_pos_void);plt.show()
#======================================================================================