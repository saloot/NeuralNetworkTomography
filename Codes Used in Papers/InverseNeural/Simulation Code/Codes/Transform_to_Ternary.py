#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
import time
import numpy as np
import sys,getopt,os
from scipy.cluster.vq import kmeans,whiten,kmeans2,vq

from CommonFunctions.auxiliary_functions import combine_weight_matrix,generate_file_name
from CommonFunctions.auxiliary_functions_digitize import caculate_accuracy,beliefs_to_ternary,parse_commands_ternary_algo
from CommonFunctions.Neurons_and_Networks import *

os.system('clear')                                              # Clear the commandline window
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:K:C:Y:U:Z:O:")

frac_stimulated_neurons,T_max,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,ternary_mode,file_name_base_results,inference_method,sparsity_flag,we_know_topology,beta,alpha0,infer_itr_max,T_range,plot_flags = parse_commands_ternary_algo(input_opts)
#==============================================================================

#================================INITIALIZATIONS===============================

#--------------------------Initialize the Network------------------------------
#Network = NeuralNet(no_layers,n_exc_array,n_inh_array,connection_prob_matrix,delay_max_matrix,random_delay_flag,'')
Network = NeuralNet(None,None,None,None,None,None,None, 'command_line',input_opts,args)
#------------------------------------------------------------------------------    

#------Calculate the Range to Assess the Effect of Recording Duration------        
if not T_range:
    T_step = int(T_max/6.0)
    T_range = range(T_step, T_max+1, T_step)
#--------------------------------------------------------------------------

#-----------------------Set Simulation Variables------------------------
adj_fact_exc = 0.75 # This is the adjustment factor for clustering algorithms (between 0 and infinity).
                    # The larger this value is (bigger than 1), the harder it would be to classify a neuron as excitatory
adj_fact_inh = 0.5  # This is the adjustment factor for clustering algorithms (between 0 and infinity).
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

#..............................................................................

#..............................................................................

#------------------------------------------------------------------------------

#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
for ensemble_count in range(ensemble_count_init,ensemble_size):    
    
    first_flag2 = 1
    
    #--------------------------READ THE NETWORK--------------------------------
    Network.read_weights(ensemble_count,file_name_base_data)
    if not Network.read_weights(ensemble_count,file_name_base_data):
        print 'Error! No inferred graph found :('
        sys.exit()
    #--------------------------------------------------------------------------        
    
    #-------------------------IN-LOOP INITIALIZATIONS--------------------------    
    if we_know_topology.lower() == 'y':
        recall_tot = {}
        prec_tot = {}    
        for l_in in range(0,Network.no_layers):
            for l_out in range(l_in,Network.no_layers):                    
                ind = str(l_in) + '_' + str(l_out)        
                recall_tot[ind] = np.zeros([len(T_range),3])
                prec_tot[ind] = np.zeros([len(T_range),3])
                
    else:
        recall_tot = {}
        prec_tot = {}
        for l_out in range(0,Network.no_layers):
            ind = str(l_out)
            recall_tot[ind] = np.zeros([len(T_range),3])
            prec_tot[ind] = np.zeros([len(T_range),3])
    #--------------------------------------------------------------------------
     
    #---------------------------CALCULATE ACCURACY-----------------------------       
    itr_T = 0
    for T in T_range:        
        if we_know_topology.lower() == 'y':
            
            print '======RESULTS IN LAYER: l_in=%s and l_out=%s======' %(l_in,l_out)
        
            for l_in in range(0,Network.no_layers):
                
                n_exc = Network.n_exc_array[l_in]
                n_inh = Network.n_inh_array[l_in]
                n = float(n_exc + n_inh)
                for l_out in range(l_in,Network.no_layers):
                    ind = str(l_in) + str(l_out)
                    pdb.set_trace()
                    temp_list = Network.Neural_Connections[ind]
                    W = temp_list[0]
                
                    #-------------------------Read the Belief Matrices-------------------------                    
                    file_name_ending_base = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)                
                    file_name_ending = generate_file_name(file_name_ending_base,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')
                    file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
                    W_inferred = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    #--------------------------------------------------------------------------
                    
                    #----------------------Adjust Algorithm's Parameters-----------------------
                    n,m = W.shape
                    fixed_entries = np.zeros([n,m])
                    W_inferred = W_inferred[:,0:m]
                    if ternary_mode == 4:
                        params[2] = fixed_entries
                    elif ternary_mode == 2:
                        P = Network.connection_prob_matrix[l_in,l_out]
                        p_exc = P * (n_exc/n)
                        p_inh = P * (n_inh/n)
                        params=[p_exc,p_inh]
                        
                    W_inferred = W_inferred/float(abs(W_inferred).max()+0.00001)
                    W_inferred = W_inferred + np.random.rand(n,m)/100000                    # This is done to keep the whitening process screwing up all-zero columns
                    W_inferred = whiten(W_inferred)
                    #--------------------------------------------------------------------------
                    
                    #-----------------Calculate the Binary Matrix From Beliefs-----------------
                    W_binary,centroids = beliefs_to_ternary(ternary_mode,10*W_inferred,params,dale_law_flag)                    
                    #--------------------------------------------------------------------------
                    
                    #--------------------------Store the Binary Matrices-----------------------
                    file_name_ending2 = file_name_ending + "_%s" %str(adj_fact_exc)
                    file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                    file_name_ending2 = file_name_ending2 + "_B_%s" %str(ternary_mode)
                
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
                    recal,precision = caculate_accuracy(W_binary,W)
                    ind = str(l_in) + '_' + str(l_out)
                    temp1 = recall_tot[ind]
                    temp1[itr_T,:] = recal
                    recall_tot[ind] = temp1
                    
                    temp1 = prec_tot[ind]
                    temp1[itr_T,:] = precision
                    prec_tot[ind] = temp1
                
                    print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
                    print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
                    print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
                    print '\n'
                    #--------------------------------------------------------------------------
                    
              
                    
        else:
            
            #--------------------Read the Whole Connectivity Matrix--------------------
            W_tot,DD_tot = combine_weight_matrix(Network)    
            #--------------------------------------------------------------------------           
            
            #-------------------------Read the Belief Matrices-------------------------
            file_name_ending = generate_file_name(Network.file_name_ending,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')
            file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending            
            W_inferred = np.genfromtxt(file_name, dtype=None, delimiter='\t')            
            #--------------------------------------------------------------------------

            #-----------------Calculate the Binary Matrix From Beliefs-----------------
            ind_this_layer = 0
            
            for l_out in range(0,Network.no_layers):
                ind = str(l_out)
                n_o = Network.n_exc_array[l_out] + Network.n_inh_array[l_out]
                
                W_temp = W_tot[:,ind_this_layer:ind_this_layer + n_o]
                
                n,m = W_temp.shape
                fixed_entries = np.zeros([n,m])
                
                W_inferred_temp = W_inferred[:,ind_this_layer:ind_this_layer + n_o]
                W_inferred_temp = W_inferred_temp[:,0:m]
                
                if ternary_mode == 4:
                    params[2] = fixed_entries
                elif ternary_mode == 2:
                    P = np.sum(abs(W_temp)>0)/float(m*n)
                    p_exc = P * (np.sum(W_temp>0)/(np.sum(abs(W_temp)>0)+0.00001))
                    p_inh = P * (np.sum(W_temp<0)/(np.sum(abs(W_temp)>0)+0.00001))
                    params=[p_exc,p_inh]
                
                W_inferred_temp = W_inferred_temp/float(abs(W_inferred_temp).max()+0.00001)
                W_inferred_temp = W_inferred_temp + np.random.rand(n,m)/100000
                W_inferred_temp = whiten(W_inferred_temp)
                
                W_binary,centroids = beliefs_to_ternary(ternary_mode,10*W_inferred_temp,params,dale_law_flag)                
                #--------------------------------------------------------------------------
        
                #--------------------------Store the Binary Matrices-----------------------
                file_name_ending2 = file_name_ending + "_" + str(l_out)
                file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
                file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                file_name_ending2 = file_name_ending2 + "_B_%s" %str(ternary_mode)
                
                
                file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2
                np.savetxt(file_name,W_binary,'%d',delimiter='\t',newline='\n')
        
                
                file_name = file_name_base_results + "/Inferred_Graphs/Scatter_Beliefs_%s.txt" %file_name_ending2                
                ww = W_inferred_temp.ravel()
                ww = np.vstack([ww,np.zeros([len(ww)])])
                np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')

                if (ternary_mode == 4):
                    file_name = file_name_base_results + "/Inferred_Graphs/Centroids_%s.txt" %file_name_ending2
                    centroids = np.vstack([centroids,np.zeros([3])])
                    np.savetxt(file_name,centroids,'%f',delimiter='\t')
                #--------------------------------------------------------------------------
                
                #---------Calculate and Display Recall & Precision for Our Method----------    
                recal,precision = caculate_accuracy(W_binary,W_temp)
                recall_tot[ind][itr_T,:] = recal
                prec_tot[ind][itr_T,:] = precision
                
                print '-------------Our method performance in ensemble %d & T = %d------------' %(ensemble_count,T)
                print 'Rec_+:   %f      Rec_-:  %f      Rec_0:  %f' %(recal[0],recal[1],recal[2])
                print 'Prec_+:  %f      Prec_-: %f      Prec_0: %f' %(precision[0],precision[1],precision[2])
                print '\n'
                #--------------------------------------------------------------------------
            
                ind_this_layer = ind_this_layer + n_o
                
        itr_T = itr_T + 1
    #======================================================================================
        
        
    #==================================SAVE THE RESULTS====================================
    T_range = np.divide(T_range,1000.0).astype(int)
    if we_know_topology.lower() == 'y':    
        for l_in in range(0,Network.no_layers):            
            for l_out in range(l_in+1,Network.no_layers):
                recal = recall_tot[ind]
                precs = prec_tot[ind]
                
                temp_ending = file_name_ending2.replace("_T_%s" %str(T),'')
                file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %temp_ending
                np.savetxt(file_name,np.vstack([T_range,recal.T]).T,'%f',delimiter='\t',newline='\n')
        
                file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %temp_ending
                np.savetxt(file_name,np.vstack([T_range,precs.T]).T,'%f',delimiter='\t',newline='\n')
        
                #file_name_ending2 = file_name_ending2.replace('_l_' + str(l_in) + '_to_' + str(l_out),'')        
                #file_name = file_name_base_results + "/Accuracies/Rec_Layers_Segregated_%s.txt" %file_name_ending2
                #if (first_flag2):
                #    acc_file = open(file_name,'w')
                #else:
                #    acc_file = open(file_name,'a')
            
                #acc_file.write("%s \t %f \t %f \n" %(ind,recal[itr_T,0],recal[itr_T,1]))
                #acc_file.close()
        
                #file_name = file_name_base_results + "/Accuracies/Prec_Layers_Segregated_%s.txt" %file_name_ending2

                #if (first_flag2):
                #    acc_file = open(file_name,'w')
                #else:
                #    acc_file = open(file_name,'a')
                #    if ( (l_in == Network.no_layers-1) and (l_out == Network.no_layers-1) ):
                #        first_flag2 = 0
            
                #acc_file.write("%s \t %f \t %f \n" %(ind,prec_tot[itr_T,0],prec_tot[itr_T,1]))
                #acc_file.close()
    else:
        for l_out in range(0,Network.no_layers):
            ind = str(l_out)
            
            file_name_ending2 = file_name_ending + "_" + str(l_out)
            file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
            file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
            file_name_ending2 = file_name_ending2 + "_B_%s" %str(ternary_mode)
                
            temp_ending = file_name_ending2.replace("_T_%s" %str(T),'')
            
            file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %temp_ending
            np.savetxt(file_name,np.vstack([T_range,recall_tot[ind].T]).T,'%f',delimiter='\t',newline='\n')
        
            file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %temp_ending
            np.savetxt(file_name,np.vstack([T_range,prec_tot[ind].T]).T,'%f',delimiter='\t',newline='\n')
        #==================================================================================
        
    #raw_input("Press a key to continue...")    
        