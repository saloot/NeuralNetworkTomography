#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
import time
import numpy as np
import sys,getopt,os
from scipy.cluster.vq import kmeans,whiten,kmeans2,vq

#from CommonFunctions.auxiliary_functions import combine_weight_matrix,generate_file_name
from CommonFunctions.auxiliary_functions_digitize import caculate_accuracy,beliefs_to_ternary,parse_commands_ternary_algo
#from CommonFunctions.Neurons_and_Networks import *

os.system('clear')                                              # Clear the commandline window
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:K:C:Y:U:Z:o:N:")

file_name_ending,file_name_base_results,ternary_mode,Var1_range,var_name,neuron_range,file_name_ground_truth,n = parse_commands_ternary_algo(input_opts)
#==============================================================================

#================================INITIALIZATIONS===============================

no_hidden_neurons = 50

#-----------------------Set Simulation Variables------------------------
adj_fact_exc = 1. # This is the adjustment factor for clustering algorithms (between 0 and infinity).
                    # The larger this value is (bigger than 1), the harder it would be to classify a neuron as excitatory
adj_fact_inh = 1  # This is the adjustment factor for clustering algorithms (between 0 and infinity).
                    # The larger this value is (bigger than 1), the harder it would be to classify a neuron as inhibitory
dale_law_flag = 0   # If 1, the ternarification algorithm returns a matrix in which the non-zero entries in a row (i.e. outgoing neural connections) have the same sign
#------------------------------------------------------------------------------

#---------------------Initialize Ternarification Parameters--------------------

#.............................Sorting-Based Approach...........................
if ternary_mode == 2:    
    params = []                     # Parameters will be set later
#..............................................................................

#..........................Clustering-based Approach...........................
if ternary_mode == 4:
    params = [adj_fact_exc,adj_fact_inh,[]]
#..............................................................................

#------------------------------------------------------------------------------

#---------------------Read The Actual Grapgh If Possible-----------------------
#file_name = '../Data/Graphs/Moritz_Actual_Connectivity.txt'
#file_name = '../Results/Inferred_Graphs/W_Pll_Moritz_I_7_S_5_T_75000_0.txt'
if file_name_ground_truth:
    W = np.genfromtxt(file_name_ground_truth, dtype=None, delimiter='\t')
    W = W.T
    n,m = W.shape
    W_ss = W[:,neuron_range]
    W_s = np.zeros([n-no_hidden_neurons,len(neuron_range)])
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
recal_exc = np.zeros([len(Var1_range)])
std_recal_exc = np.zeros([len(Var1_range)])
prec_exc = np.zeros([len(Var1_range)])
std_prec_exc = np.zeros([len(Var1_range)])
    
recal_void = np.zeros([len(Var1_range)])
std_recal_void = np.zeros([len(Var1_range)])
prec_void = np.zeros([len(Var1_range)])
std_prec_void = np.zeros([len(Var1_range)])

recal_inh = np.zeros([len(Var1_range)])
std_recal_inh = np.zeros([len(Var1_range)])
prec_inh = np.zeros([len(Var1_range)])
std_prec_inh = np.zeros([len(Var1_range)])
#------------------------------------------------------------------------------

    
#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
itr_V1 = 0
W_inferred = np.zeros([n-no_hidden_neurons,len(neuron_range)])
W_infer = np.zeros([len(file_name_ending_list),n-no_hidden_neurons])


for v1 in Var1_range:
    
    exec('%s = v1' %var_name)
    
    #------------------------Read the Inferred Weights-------------------------
    itr_n = 0
    for n_ind in neuron_range:
        
        W_infer = 0*W_infer
        itr_i = 0
        for file_name_ending in file_name_ending_list:
            temp = file_name_ending.split('_')
            
            ind1 = temp.index(var_name[0])
            temp[ind1+1] = str(eval(plot_vars[0]))
            
            file_name_ending_mod = '_'.join(temp)
    
            file_name = "Inferred_Graphs/W_Pll_%s_%s.txt" %(file_name_ending_mod,str(n_ind))
    
            file_name = file_name_base_results + '/' + file_name
            
            W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            W_infer[itr_i,:] = W_read[0:n-no_hidden_neurons]
        
            #if itr_i:
            #    ww = sum(W_infer[0:itr_i+1,:,itr_n])/float(itr_i+1)
            #    #W = W_infer[itr_i,:]
            #else:
            #    ww = W_infer[itr_i,:]
            itr_i = itr_i + 1
    
        if itr_i > 1:
            W_inferred[0:min(m-no_hidden_neurons,len(W_read)),itr_n] = W_infer[0:itr_i,:].mean(axis = 0)
        else:
            W_inferred[0:min(m-no_hidden_neurons,len(W_read)),itr_n] = W_infer[itr_i-1,:]

        
        if no_hidden_neurons:
            file_name_hidden = "Inferred_Graphs/Hidden_Neurons_%s_%s.txt" %(file_name_ending_mod,str(n_ind))
            file_name = file_name_base_results + '/' + file_name_hidden
            hidden_neurons = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            W_h = np.delete(W_ss[:,n_ind],hidden_neurons,0)
            W_s[:,itr_n] = W_h
        else:
            W_s[:,itr_n] = W_ss[:,n_ind]
            
        itr_n = itr_n + 1
    
    W_inferred_s = W_inferred#[:,neuron_range]
    #W_inferred_s = W_inferred_s[:-1]
    #--------------------------------------------------------------------------

    #-----------------Calculate the Binary Matrix From Beliefs-----------------
    W_binary,centroids = beliefs_to_ternary(ternary_mode,W_inferred_s,params,dale_law_flag)                    
    #--------------------------------------------------------------------------
    
    #--------------------------Store the Binary Matrices-----------------------
    file_name_ending2 = file_name_ending_mod + "_%s" %str(adj_fact_exc)
    file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
    file_name_ending2 = file_name_ending2 + "_B_%s" %str(ternary_mode)
    file_name_ending2 = file_name_ending2 + "_n_%s_%s" %(str(neuron_range[0]),str(neuron_range[-1]))
                
    file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2
    np.savetxt(file_name,W_binary,'%d',delimiter='\t',newline='\n')
    
                
    file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_%s.txt" %file_name_ending2                
    ww = W_inferred.ravel()
    ww = np.vstack([ww,np.zeros([len(ww)])])
    np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')

    if (ternary_mode == 4):
        file_name = file_name_base_results + "/Inferred_Graphs/Centroids_%s.txt" %file_name_ending2
        centroids = np.vstack([centroids,np.zeros([3])])
        np.savetxt(file_name,centroids,'%f',delimiter='\t')
    #--------------------------------------------------------------------------
                    
    #---------Calculate and Display Recall & Precision for Our Method----------    
    if file_name_ground_truth:    
        recal,precision = caculate_accuracy(W_binary,W_s)
        
        recal_exc[itr_V1] = recal[0]
        recal_inh[itr_V1] = recal[1]
        recal_void[itr_V1] = recal[2]
        
        prec_exc[itr_V1] = precision[0]
        prec_inh[itr_V1] = precision[1]
        prec_void[itr_V1] = precision[2]
        
    
    itr_V1 = itr_V1 + 1
#======================================================================================

#==============================CALCULATE THE ROC CURVE=================================
if file_name_ground_truth:
    m,n = W.shape
    
    #---------------------------For Excitatory Connections-----------------------------
    if len(neuron_range) == 1:
        W_inferred_s = np.reshape(W_inferred_s,[len(W_inferred_s),1])
    itr_n = 0
    for n_ind in neuron_range:
        ww = W_inferred_s[:,itr_n]
        ww = ww - ww.mean()
        ww = whiten(ww)
        W_inferred_s[:,itr_n] = ww
    
        itr_n = itr_n + 1
        
        
    min_val = int(100*W_inferred_s.min())
    max_val = int(100*W_inferred_s.max())
    mid_val = int(0.5*(max_val+min_val))
    
    
    
    val_range_exc = range(0,max_val)
    val_range_exc = np.array(val_range_exc)/100.0
    val_range_inh = range(0,min_val,-1)
    val_range_inh = np.array(val_range_inh)/100.0

    true_pos_exc = np.zeros([len(val_range_exc)])
    false_pos_exc = np.zeros([len(val_range_exc)])
    true_pos_inh = np.zeros([len(val_range_inh)])
    false_pos_inh = np.zeros([len(val_range_inh)])
    true_pos_void = np.zeros([min(len(val_range_inh),len(val_range_exc))])
    false_pos_void = np.zeros([min(len(val_range_inh),len(val_range_exc))])
    
    itr_n = 0
    for n_ind in neuron_range:
        itr_V1 = 0
        
        ww = W_inferred_s[:,itr_n]
        W_temp = ww#[:-1]    
        for thresh in val_range_exc:
            
            #~~~~~~~~Calcualte True Positive and False Positive Rates~~~~~~~~~
            W_ter = (W_temp>=thresh).astype(int)
            
            true_pos_exc[itr_V1] = true_pos_exc[itr_V1] + (sum(np.multiply((W_ter>0).astype(int),(W_s[:,n_ind]>0).astype(int))))/float(sum(W_s[:,itr_n]>0))
            false_pos_exc[itr_V1] = false_pos_exc[itr_V1] + sum(np.multiply((W_ter>0).astype(int),(W_s[:,n_ind]<=0).astype(int)))/float(sum(W_s[:,itr_n]<=0))
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            itr_V1 = itr_V1 + 1
        
        itr_V1 = 0
        ww = W_inferred_s[:,itr_n]
        W_temp = ww#[:-1]
        
    
    
        for thresh in val_range_inh:
            
            #~~~~~~~~Calcualte True Positive and False Positive Rates~~~~~~~~~
            W_ter = (W_temp<=thresh).astype(int)
            
            true_pos_inh[itr_V1] = true_pos_inh[itr_V1] + (sum(np.multiply((W_ter>0).astype(int),(W_s[:,n_ind]<0).astype(int))))/float(sum(W_s[:,itr_n]<0))
            false_pos_inh[itr_V1] = false_pos_inh[itr_V1] + sum(np.multiply((W_ter>0).astype(int),(W_s[:,n_ind]>=0).astype(int)))/float(sum(W_s[:,itr_n]>=0))
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            itr_V1 = itr_V1 + 1
            
        itr_V1 = 0
        ww = W_inferred_s[:,itr_n]
        W_temp = ww#[:-1]
        
        for itr_V1 in range(0,min(len(val_range_inh),len(val_range_exc))):
            
            thr1 = val_range_inh[itr_V1]
            thr2 = val_range_exc[itr_V1]
            
            #~~~~~~~~Calcualte True Positive and False Positive Rates~~~~~~~~~
            W_ter = (W_temp>=thr2).astype(int) + (W_temp<=thr1).astype(int)
            W_ter = (W_ter == 0).astype(int)
            
            true_pos_void[itr_V1] = true_pos_void[itr_V1] + (sum(np.multiply((W_ter>0).astype(int),(W_s[:,n_ind]==0).astype(int))))/float(sum(W_s[:,itr_n]==0))
            false_pos_void[itr_V1] = false_pos_void[itr_V1] + sum(np.multiply((W_ter>0).astype(int),(W_s[:,n_ind]!=0).astype(int)))/float(sum(W_s[:,itr_n]!=0))
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            
        itr_n = itr_n + 1
    
    true_pos_exc = true_pos_exc/float(len(neuron_range))
    false_pos_exc = false_pos_exc/float(len(neuron_range))
    true_pos_inh = true_pos_inh/float(len(neuron_range))
    false_pos_inh = false_pos_inh/float(len(neuron_range))
    true_pos_void = true_pos_void/float(len(neuron_range))
    false_pos_void = false_pos_void/float(len(neuron_range))
    #----------------------------------------------------------------------------------        
    
    
    #----------------------------------------------------------------------------------
    val_range = range(0,100)
    val_range = np.array(val_range)/100.0
    plt.plot(false_pos_exc,true_pos_exc);plt.plot(val_range,val_range,'r');plt.show()
    plt.plot(false_pos_inh,true_pos_inh);plt.show()#plt.plot(val_range,val_range,'r');plt.show()
    plt.plot(false_pos_void,true_pos_void);plt.show()
    
    
    file_name = file_name_base_results + "/Plot_Results/ROC_exc_%s.txt" %file_name_ending
    np.savetxt(file_name,np.vstack([false_pos_exc,true_pos_exc]).T,'%f',delimiter='\t',newline='\n')
    
    file_name = file_name_base_results + "/Plot_Results/ROC_inh_%s.txt" %file_name_ending
    np.savetxt(file_name,np.vstack([false_pos_inh,true_pos_inh]).T,'%f',delimiter='\t',newline='\n')
    
    file_name = file_name_base_results + "/Plot_Results/ROC_void_%s.txt" %file_name_ending
    np.savetxt(file_name,np.vstack([false_pos_void,true_pos_void]).T,'%f',delimiter='\t',newline='\n')
#======================================================================================


#=============PERFORM THE TERNARIFICATION ON THE EFFECT OF HIDDEN NEURONS==============
itr_V1 = 0
W_inferred = np.zeros([n-no_hidden_neurons,len(neuron_range)])
W_infer = np.zeros([len(file_name_ending_list),n-no_hidden_neurons])

no_hidden_neurons_list = [0,50,100,150,200,250,300,350,400,450,500]

    
prec_exc_h = np.zeros([len(no_hidden_neurons_list)])
prec_inh_h = np.zeros([len(no_hidden_neurons_list)])    
prec_void_h = np.zeros([len(no_hidden_neurons_list)])

rec_exc_h = np.zeros([len(no_hidden_neurons_list)])
rec_inh_h = np.zeros([len(no_hidden_neurons_list)])    
rec_void_h = np.zeros([len(no_hidden_neurons_list)])


itr_V1 = 0
for no_hidden_neurons in no_hidden_neurons_list:
        
    W_inferred = np.zeros([n-no_hidden_neurons,m])
    prec_vector = np.zeros([3,len(neuron_range)])
    rec_vector = np.zeros([3,len(neuron_range)])
    itr_n = 0
    for n_ind in neuron_range:
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~~~~~~~~~
        if no_hidden_neurons:
            temp = file_name_ending_hid.split('_')
        else:
            temp = file_name_ending.split('_')
                
        if ('I') in plot_vars:                
            ind1 = temp.index('I')
            temp[ind1+1] = str(inference_method)
            
        ind1 = temp.index(plot_vars[0])
        temp[ind1+1] = str(eval(plot_vars[0]))
        try:
            ind1 = temp.index(plot_vars[1])
            temp[ind1+1] = str(eval(plot_vars[1]))
        except:
            ttemp = 1
            
        file_name_ending_mod = '_'.join(temp)
            
        file_name_ending_mod = file_name_ending_mod + '_' + str(n_ind)
            
        if no_hidden_neurons:
            #file_name_ending = file_name_ending + '_F_' + str(int(no_hidden_neurons)) + id_generator()
            file_name = "Inferred_Graphs/W_Pll_%s_F_%s*" %(file_name_ending_mod,str(no_hidden_neurons))
            file_name_hidden = "Inferred_Graphs/Hidden_Neurons_%s_F_%s*" %(file_name_ending_mod,str(no_hidden_neurons))
        else:
            file_name = "Inferred_Graphs/W_Pll_%s.txt" %(file_name_ending_mod)
            
        #----------------Read All Instances Generated for the Same Ensemble---------------
        file_name_list = []
        file_name_resources_list = []
        file_name_hidden_list = []
            
        if no_hidden_neurons:
            file_name_s = "W_Pll_%s_F_%s" %(file_name_ending_mod,str(no_hidden_neurons))
                
            for f_name in os.listdir(file_name_base_results+'/Inferred_Graphs/'):
                if (f_name.startswith(file_name_s)) and ('_n_' not in f_name):

                    file_name_e = f_name[-10:-4]
                    file_name_list.append(file_name_base_results + '/Inferred_Graphs/' + file_name_s + file_name_e + '.txt')
                    file_name_hidden_list.append(file_name_base_results + '/Inferred_Graphs/Hidden_Neurons_' + file_name_ending_mod + '_F_' + str(no_hidden_neurons) + file_name_e + '.txt')
                
        #---------------------------------------------------------------------------------
            
        else:
            file_name_list=[file_name_base_results + '/' + file_name]
            
        for il in range(0,len(file_name_list)):
            file_name = file_name_list[il]
                
            if no_hidden_neurons:
                file_name_hidden = file_name_hidden_list[il]
                    
            try:
                W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                #W_read = W_read/np.abs(W_read[:-1]).max()
                W_inferred[0:min(m-no_hidden_neurons,len(W_read)),n_ind] = W_read[0:min(m-no_hidden_neurons,len(W_read))]
                    
                        
                if no_hidden_neurons:
                    hidden_neurons = np.genfromtxt(file_name_hidden, dtype=None, delimiter='\t')
                        
                print "Got the file!"
                
            except:
                found_file_tot = 0
                print 'Sorry I can not find the corresponding inference file for the netwrok'
                pdb.set_trace()
            
            #-----------------Calculate the Binary Matrix From Beliefs-----------------
            W_inferred_s = W_inferred[:,neuron_range]
            W_s = W_read[0:min(m-no_hidden_neurons,len(W_read))]
            W_binary,centroids = beliefs_to_ternary(ternary_mode,W_s,params,dale_law_flag)
            
            #--------------------------------------------------------------------------
    
            if  file_name_ground_truth:                                        
                W_r = np.reshape(W[:,n_ind],[len(W[:,n_ind]),1])
                
                    
                    
                if no_hidden_neurons:
                    W_r = np.delete(W_r,hidden_neurons,0)
                    
                W_s = np.reshape(W_s,[len(W_s),1])
                W_r = np.reshape(W_r,[len(W_r),1])
                
                recal,precision = caculate_accuracy(W_binary,W_r)
        
                prec_vector[0,itr_n] = prec_vector[0,itr_n] + precision[0]
                prec_vector[1,itr_n] = prec_vector[1,itr_n] + precision[1]
                prec_vector[2,itr_n] = prec_vector[2,itr_n] + precision[2]
                
                rec_vector[0,itr_n] = rec_vector[0,itr_n] + recal[0]
                rec_vector[1,itr_n] = rec_vector[1,itr_n] + recal[1]
                rec_vector[2,itr_n] = rec_vector[2,itr_n] + recal[2]
                        
        #~~~~~~~~~~~~~~~~~~~~~~~~Update the Variables~~~~~~~~~~~~~~~~~~~~~~~~~~
        prec_exc_h[itr_V1] = prec_vector[0,itr_n]/float(len(file_name_list))
        prec_inh_h[itr_V1] = prec_vector[1,itr_n]/float(len(file_name_list))
        prec_void_h[itr_V1] = prec_vector[2,itr_n]/float(len(file_name_list))
            
        rec_exc_h[itr_V1] = rec_vector[0,itr_n]/float(len(file_name_list))
        rec_inh_h[itr_V1] = rec_vector[1,itr_n]/float(len(file_name_list))
        rec_void_h[itr_V1] = rec_vector[2,itr_n]/float(len(file_name_list))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
        itr_n = itr_n + 1        
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
    itr_V1 = itr_V1 + 1
    
no_hidden_neurons_list = np.array(no_hidden_neurons_list)
no_hidden_neurons_list_str = no_hidden_neurons_list.astype('str')
temp_ending = file_name_ending + '_' + '_'.join(no_hidden_neurons_list_str)
file_name = file_name_base_results + "/Plot_Results/Prec_Reca_All_Effect_hidden_%s.txt" %(temp_ending)

hidden_visible_ration = np.divide(no_hidden_neurons_list,(n-no_hidden_neurons_list).astype(float))
temp = np.vstack([prec_exc_h,prec_inh_h,prec_void_h,rec_exc_h,rec_inh_h,rec_void_h])
temp = temp.T
temp = np.hstack([np.reshape(hidden_visible_ration,[len(hidden_visible_ration),1]),temp])

np.savetxt(file_name,temp,'%f',delimiter='\t',newline='\n')




#==================================SAVE THE RESULTS====================================
if var_name == 'T':
    Var1_range = np.divide(Var1_range,1000.0).astype(int)

temp_ending = file_name_ending2.replace("_%s_%s" %(var_name,str(v1)),'')


temp = np.vstack([recal_exc,recal_inh,recal_void])
temp = temp.T
temp = np.hstack([np.reshape(Var1_range,[len(Var1_range),1]),temp])
file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %temp_ending
np.savetxt(file_name,temp,'%f',delimiter='\t',newline='\n')
        
file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %temp_ending
temp = np.vstack([prec_exc,prec_inh,prec_void])
temp = temp.T
temp = np.hstack([np.reshape(Var1_range,[len(Var1_range),1]),temp])

np.savetxt(file_name,temp,'%f',delimiter='\t',newline='\n')
