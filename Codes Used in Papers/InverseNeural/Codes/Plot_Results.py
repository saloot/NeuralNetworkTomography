#=======================IMPORT THE NECESSARY LIBRARIES=========================
import numpy as np
import sys,getopt,os
from time import time
import matplotlib.pyplot as plt
import copy
from scipy.cluster.vq import whiten

from CommonFunctions.Neurons_and_Networks import *
from CommonFunctions.default_values import *
from CommonFunctions.auxiliary_functions_digitize import caculate_accuracy,parse_commands_ternary_algo
from CommonFunctions.auxiliary_functions import generate_file_name,save_plot_results,combine_weight_matrix,calculate_belief_quality

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:J:K:U:Z:")
frac_stimulated_neurons,T_max,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,ternary_mode,file_name_base_results,inference_method,sparsity_flag,we_know_topology,beta,alpha0,infer_itr_max = parse_commands_ternary_algo(input_opts)
#==============================================================================

#================================INITIALIZATIONS===============================

#--------------------------Initialize the Network------------------------------
Network = NeuralNet(None,None,None,None,None,None,None, 'command_line',input_opts,args)
#------------------------------------------------------------------------------    

#------Calculate the Range to Assess the Effect of Recording Duration------        
T_step = int(T_max/6.0)
T_range = range(T_step, T_max+1, T_step)
#T_range = [2779,5458,8137,10816]
#--------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
whiten_flag = 1                     # If 1, the algorithm whitens the inferred graph before calculating the results
zero_diagonals_flag = 1             # If 1, the diagonal elements (self feedback-loops) will be set to 0 before calculating belief qualities
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
std_exc = {};std_inh = {};std_void = {};mean_void_r = {}
mean_exc = {};mean_inh = {};mean_void = {};std_void_r = {};


for l_out in range(0,Network.no_layers):    
    n_layer = Network.n_exc_array[l_out]+Network.n_inh_array[l_out]
    if we_know_topology.lower() == 'n':
        ind = str(l_out)
        std_exc[ind] = np.zeros([len(T_range),n_layer])
        std_inh[ind] = np.zeros([len(T_range),n_layer])
        std_void[ind] = np.zeros([len(T_range),n_layer])
        mean_exc[ind] = np.zeros([len(T_range),n_layer])
        mean_inh[ind] = np.zeros([len(T_range),n_layer])
        mean_void[ind] = np.zeros([len(T_range),n_layer])
        std_void_r[ind] = np.zeros([len(T_range),n_layer])
        mean_void_r[ind] = np.zeros([len(T_range),n_layer])
    else:
        for l_in in range (0,l_out + 1):
            ind = str(l_in) + str(l_out)
            ind = str(l_out)
            std_exc[ind] = np.zeros([len(T_range),n_layer])
            std_inh[ind] = np.zeros([len(T_range),n_layer])
            std_void[ind] = np.zeros([len(T_range),n_layer])
            mean_exc[ind] = np.zeros([len(T_range),n_layer])
            mean_inh[ind] = np.zeros([len(T_range),n_layer])
            mean_void[ind] = np.zeros([len(T_range),n_layer])

#------------------------------------------------------------------------------

#==============================================================================


#=============================READ THE RESULTS=================================
for ensemble_count in range(0,ensemble_size):
    
    #-----------------------------Read The Weights-----------------------------    
    Network.read_weights(ensemble_count,file_name_base_data)
    
    if we_know_topology.lower() != 'y':
        W,DD_tot = combine_weight_matrix(Network)
    #--------------------------------------------------------------------------
    
    
    #-------------Extract and Process the Data from Stored Files---------------
    itr_T = 0
    
    for T in T_range:
        if we_know_topology.lower() != 'y':
            #~~~~~~~~~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~~~~~~~~~
            file_name_ending = generate_file_name(Network.file_name_ending,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')                    
            file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
            W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')                    
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        for l_out in range(0,Network.no_layers):
            if we_know_topology.lower() == 'y':
                for l_in in range(0,l_out+1):
        
                    #~~~~~~~~~~~~~~~Extract the Weights for Each Layer~~~~~~~~~~~~~~~~
                    ind = str(l_in) + str(l_out)
                    temp_list = Network.Neural_Connections[ind]
                    W = temp_list[0]
                    n,m = W.shape
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    #~~~~~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~~~~~
                    file_name_ending_base = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)                
                    file_name_ending = generate_file_name(file_name_ending_base,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')                    
                    file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending                    
                    W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~Claculate Belief Quality~~~~~~~~~~~~~~~~~~
                    means_vector,std_vector = calculate_belief_quality(Network,W_inferred_our,W,l_out,whiten_flag,zero_diagonals_flag,we_know_topology)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~~Update the Variables~~~~~~~~~~~~~~~~~~~~~                    
                    mean_exc[ind][itr_T,:] = mean_exc[ind][itr_T,:] + means_vector[0]
                    std_exc[ind][itr_T,:] = std_exc[ind][itr_T,:] + std_vector[0]
                    
                    mean_inh[ind][itr_T,:] = mean_inh[ind][itr_T,:] + means_vector[1]
                    std_inh[ind][itr_T,:] = std_inh[ind][itr_T,:] + std_vector[1]
                    
                    mean_void[ind][itr_T,:] = mean_void[ind][itr_T,:] + means_vector[2]
                    std_void[ind][itr_T,:] = std_void[ind][itr_T,:] + std_vector[2]
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:                    
                    
                #~~~~~~~~~~~~~~~~~Claculate Belief Quality~~~~~~~~~~~~~~~~~~
                means_vector,std_vector = calculate_belief_quality(Network,W_inferred_our,W,l_out,whiten_flag,zero_diagonals_flag,we_know_topology)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                #~~~~~~~~~~~~~~~~~~~~Update the Variables~~~~~~~~~~~~~~~~~~~~~~~
                ind = str(l_out)
                mean_exc[ind][itr_T,:] = mean_exc[ind][itr_T,:] + means_vector[0]
                std_exc[ind][itr_T,:] = std_exc[ind][itr_T,:] + std_vector[0]
                
                mean_inh[ind][itr_T,:] = mean_inh[ind][itr_T,:] + means_vector[1]
                std_inh[ind][itr_T,:] = std_inh[ind][itr_T,:] + std_vector[1]
                    
                mean_void[ind][itr_T,:] = mean_void[ind][itr_T,:] + means_vector[2]
                std_void[ind][itr_T,:] = std_void[ind][itr_T,:] + std_vector[2]
                
                if we_know_topology.lower() != 'y' and Network.no_layers > 1:
                    mean_void_r[ind][itr_T,:] =  mean_void_r[ind][itr_T,:] + means_vector[3]
                    std_void_r[ind][itr_T,:] = std_void_r[ind][itr_T,:] + std_vector[3]
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        
        itr_T = itr_T + 1
    #--------------------------------------------------------------------------    

#==============================================================================


#=======================POST-PROCESS AND STORE THE RESULTS=====================
T_range = np.divide(T_range,1000.0)                 # Display the results in seconds (rather than mili seconds)
for l_out in range(0,Network.no_layers):
    if we_know_topology.lower() == 'y':
        for l_in in range(0,l_out+1):
            #~~~~~~~~~~~~~~~~~~Update the Variables~~~~~~~~~~~~~~~~~~~~~
            ind = str(l_in) + str(l_out)
            mean_exc[ind] = np.divide(mean_exc[ind],ensemble_size)
            std_exc[ind] = np.divide(std_exc[ind],ensemble_size)
                    
            mean_inh[ind] = np.divide(mean_inh[ind],ensemble_size)
            std_inh[ind] = np.divide(std_inh[ind],ensemble_size)
                    
            mean_void[ind] = np.divide(mean_void[ind],ensemble_size)
            std_void[ind] = np.divide(std_void[ind],ensemble_size)
                
            mean_exc[ind] = mean_exc[ind].mean(axis = 1)
            std_exc[ind] = std_exc[ind].mean(axis = 1)
            mean_inh[ind] = mean_inh[ind].mean(axis = 1)
            std_inh[ind] = std_inh[ind].mean(axis = 1)
            mean_void[ind] = mean_void[ind].mean(axis = 1)
            std_void[ind] = std_void[ind].mean(axis = 1)                
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #~~~~~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~~~~~
            file_name_ending_base = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)                
            file_name_ending = generate_file_name(file_name_ending_base,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')                    
            file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending                    
            W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
            #~~~~~~~~~~~~~~~~~~~Plot the Results~~~~~~~~~~~~~~~~~~~~~~~~
            plt.errorbar(T_range,mean_exc[ind],std_exc[ind],color='r',label='Excitatory');
            plt.errorbar(T_range,mean_inh[ind],std_inh[ind],color='b',label='Inhibitory');
            plt.errorbar(T_range,mean_void[ind],std_void[ind],color='g',label='Void');
            
            plt.xlabel('t(s)', fontsize=16)
            plt.ylabel('Average of beliefs', fontsize=16)
            plt.legend(loc='lower left')
            plt.show();
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #~~~~~~~~~~~~~~~Plot the Scatter of Beliefs~~~~~~~~~~~~~~~~~
            if whiten_flag:
                n,m = W_inferred_our.shape
                W_inferred_our = W_inferred_our + np.random.rand(n,m)/100000
                W_inferred_our  = whiten(W_inferred_our)
            
            plt.scatter(np.sign(W.ravel()),W_inferred_our.ravel())
            plt.xlabel('G (actual)', fontsize=16)
            plt.ylabel('W (inferred)', fontsize=16)
            plt.show()
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
            #~~~~~~~~~~~~~~~~~~~~Save the Results~~~~~~~~~~~~~~~~~~~~~~~
            save_plot_results(T_range,mean_exc[ind],std_exc[ind],mean_inh[ind],std_inh[ind],mean_void[ind],std_void[ind],mean_void_r[ind],std_void_r[ind],file_name_base_results,file_name_ending,0,W_inferred_our,W)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    else:                    
        
        #~~~~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~~~~
        file_name_ending = generate_file_name(Network.file_name_ending,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')                    
        file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
        W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')                    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        #~~~~~~~~~~~~~~~~~~~~Update the Variables~~~~~~~~~~~~~~~~~~~~~~~
        ind = str(l_out)
        mean_exc[ind] = np.divide(mean_exc[ind],ensemble_size)
        std_exc[ind] = np.divide(std_exc[ind],ensemble_size)
                    
        mean_inh[ind] = np.divide(mean_inh[ind],ensemble_size)
        std_inh[ind] = np.divide(std_inh[ind],ensemble_size)
                    
        mean_void[ind] = np.divide(mean_void[ind],ensemble_size)
        std_void[ind] = np.divide(std_void[ind],ensemble_size)
                
        mean_exc[ind] = mean_exc[ind].mean(axis = 1)
        std_exc[ind] = std_exc[ind].mean(axis = 1)
        mean_inh[ind] = mean_inh[ind].mean(axis = 1)
        std_inh[ind] = std_inh[ind].mean(axis = 1)
        mean_void[ind] = mean_void[ind].mean(axis = 1)
        std_void[ind] = std_void[ind].mean(axis = 1)   
    
        in_recurrent_flag = 0 
        if we_know_topology.lower() != 'y' and Network.no_layers > 1:
            mean_void_r[ind] = np.divide(mean_void_r[ind],ensemble_size)
            std_void_r[ind] = np.divide(std_void_r[ind],ensemble_size)
                
            mean_void_r[ind] = mean_void_r[ind].mean(axis = 1)   
            std_void_r[ind] = std_void_r[ind].mean(axis = 1)
            in_recurrent_flag = 1
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~Plot the Results~~~~~~~~~~~~~~~~~~~~~~~~
        plt.errorbar(T_range,mean_exc[ind],std_exc[ind],color='r',label='Excitatory');
        plt.errorbar(T_range,mean_inh[ind],std_inh[ind],color='b',label='Inhibitory');
        plt.errorbar(T_range,mean_void[ind],std_void[ind],color='g',label='Void');
        if Network.no_layers > 1:
            plt.errorbar(T_range,mean_void_r[ind],std_void_r[ind],color='k',label='Void, Recurrent');
        
        plt.xlabel('t(s)', fontsize=16)
        plt.ylabel('Average of beliefs', fontsize=16)
        plt.legend(loc='lower left')
        plt.show();
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
        #~~~~~~~~~~~~~~~Plot the Scatter of Beliefs~~~~~~~~~~~~~~~~~
        if whiten_flag:
            n,m = W_inferred_our.shape
            W_inferred_our = W_inferred_our + np.random.rand(n,m)/100000
            W_inferred_our  = whiten(W_inferred_our)
            
        plt.scatter(np.sign(W.ravel()),W_inferred_our.ravel())
        plt.xlabel('G (actual)', fontsize=16)
        plt.ylabel('W (inferred)', fontsize=16)
        plt.show()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~~Save the Results~~~~~~~~~~~~~~~~~~~~~~~
        file_name_ending_temp = file_name_ending + '_' + str(l_out)
        save_plot_results(T_range,mean_exc[ind],std_exc[ind],mean_inh[ind],std_inh[ind],mean_void[ind],std_void[ind],mean_void_r[ind],std_void_r[ind],file_name_base_results,file_name_ending_temp,in_recurrent_flag,W_inferred_our,W)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
#==============================================================================


#=============================CREATE SCATTER PLOTS==============================
#==============================================================================
if 0:
    T = 5458
    #T = 246
    Network.read_weights(ensemble_count,file_name_base_data)
    if we_know_topology == 'Y':
        n,m = W.shape
        file_name_ending23 = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)
    else:
        file_name_ending23 = Network.file_name_ending
        
    file_name_ending23 = file_name_ending23 + '_I_' + str(inference_method)
    file_name_ending23 = file_name_ending23 + '_Loc_' + we_know_topology
    file_name_ending23 = file_name_ending23 + '_Pre_' + pre_synaptic_method
    file_name_ending23 = file_name_ending23 + '_G_' + generate_data_mode
    file_name_ending23 = file_name_ending23 + '_X_' + str(infer_itr_max)
    file_name_ending23 = file_name_ending23 + '_Q_' + str(frac_stimulated_neurons)
    if (sparsity_flag):
        file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)
    file_name_ending2 = file_name_ending23 +"_T_%s" %str(T)
    if delay_known_flag == 'Y':
        file_name_ending2 = file_name_ending2 +"_DD_Y"
    
    file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_%s.txt" %file_name_ending2
    temp = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    m,n = temp.shape
    B_exc_void = temp[m-1,:]
    
    file_name = file_name_base_results + "/Plot_Results/Gap_mean_void_inh_%s.txt" %file_name_ending2
    temp = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    m,n = temp.shape
    B_void_inh = temp[m-1,:]
    
    #file_name = file_name_base_results + "/Mean_var_inh_%s.txt" %file_name_ending2
    #temp = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #m,n = temp.shape
    #B_inhh = temp[m-1,:]
    
    B_tot_ind = [0,1]
    B_tott = [B_void_inh[1],B_exc_void[1]]
    
    file_name = file_name_base_results + "/Plot_Results/Gap_final_e_v_i_%s.txt" %file_name_ending2
    ww = np.vstack([B_tot_ind,B_tott])
    np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')
    
    
    
if 0:
    #import igraph
    #from igraph import *
    #g = Graph()
    #g.add_vertices(n)
    n,m = W.shape
    
    acc_file = open('Adj.txt','w')
    
    eds = np.nonzero(W[43:53,43:53])
    l = len(eds[0])
    for i in range(0,l):
        #g.add_edges([(eds[0][i],eds[1][i])])
        if 1:
            node_1 = str(eds[0][i])
            node_2 = str(eds[1][i])
            acc_file.write("%s \t %s \t %s \n" %(node_1,str(np.sign(W[eds[0][i],eds[1][i]])),node_2))
    
    
    acc_file.close()
    
    
    acc_file = open('Adj_Inferred.txt','w')
    
    eds = np.nonzero(WW[43:53,43:53])
    l = len(eds[0])
    for i in range(0,l):
        #g.add_edges([(eds[0][i],eds[1][i])])
        if 1:
            node_1 = str(eds[0][i])
            node_2 = str(eds[1][i])
            acc_file.write("%s \t %s \t %s \n" %(node_1,str(np.sign(WW[eds[0][i],eds[1][i]])),node_2))
    
    
    acc_file.close()
    
    
    acc_file = open('Adj_Error.txt','w')
    
    W_e = WW - np.sign(W)
    eds = np.nonzero(W_e[43:53,43:53])
    l = len(eds[0])
    for i in range(0,l):
        #g.add_edges([(eds[0][i],eds[1][i])])
        if 1:
            node_1 = str(eds[0][i])
            node_2 = str(eds[1][i])
            acc_file.write("%s \t %s \t %s \n" %(node_1,str(np.sign(W_e[eds[0][i],eds[1][i]])),node_2))
    
    
    acc_file.close()
    
    
    
    n,m = W.shape
    
    acc_file = open('Associ_JavaS.txt','w')
    
    eds = W_inferred_our_tot[43:53,43:53]
    eds = 10*eds/(abs(eds).max())
    for i in range(0,10):
        for j in range(0,10):            
            node_1 = '{from: ' + str(i) + ','
            node_2 = 'to: ' + str(j) + ','
            wei = 'value: ' + str(eds[i,j]) + ','
            #if 
            titl = "title: 'e'},"
            acc_file.write("%s \t %s \t %s \t %s \n" %(node_1,wei,node_2,titl))
    
    
    acc_file.close()
    
    
    acc_file = open('Associ_Matr_JavaS.txt','w')
    eds = W_inferred_our_tot[43:53,43:53]
    eds = 10*eds/(abs(eds).max())
    for i in range(0,10):
        for j in range(0,10):
            entry = 'x[' + str(i) + ']'
            entry = entry + '[' + str(j) + ']='
            entry = entry + str(eds[i,j]) + ';'
            
            acc_file.write("%s \n" %(entry))
    
    
    acc_file.close()
    
    
    
    acc_file = open('Actual_JavaS.txt','w')
    
    eds = W[43:53,43:53]
    eds = 10*eds/(abs(eds).max())
    for i in range(0,10):
        for j in range(0,10):
            if abs(eds[i,j]) > 0:
                node_1 = '{from: ' + str(i) + ','
                node_2 = 'to: ' + str(j) + ','
                if eds[i,j] > 0:
                    wei = "arrows:'to', color:{color:'red'}},"
                else:
                    wei = "arrows:'to', color:{color:'blue'}},"
            
                acc_file.write("%s \t %s \t %s \n" %(node_1,node_2,wei))
    
    
    acc_file.close()
    
    #layout = g.layout_kamada_kawai()
    #layout = g.layout("kk")
    #plot(g, layout = layout)

