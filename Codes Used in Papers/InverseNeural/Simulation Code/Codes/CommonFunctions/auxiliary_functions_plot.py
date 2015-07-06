#=======================IMPORT THE NECESSARY LIBRARIES=========================
import math
#from brian import *
import numpy as np
from scipy import sparse
import pdb
import random
import copy
import numpy.ma as ma

try: 
    import plotly.plotly as pltly
    from plotly.graph_objs import *    
except:
    print 'Plotly was not found. No problem though, life goes on ;)'    
#==============================================================================


#==============================================================================
#=========================BELIEF QUALITY ASSESSMENT============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function calculates the quality of the computed beliefs about the neural
# graph

# INPUT:
#    Network: the object that contains the informatin about the connectivity pattern of each layer
#    W_inferred: the inferred graph (association matrix)
#    W_orig: the actual graph (ground truth)
#    l_out: the index of output layer (an integer)
#    whiten_flag: if 1, the inferred graph will but whitened before calculating belief qualities
#    zero_diagonals_flag: if 1, the diagonal elements (self feedback-loops) will be set to 0 before calculating belief qualities
#    we_know_topology: if 'Y', the inference has been topology-aware

# OUTPUT:
#    mean_beliefs:    the avaerage beliefs for inhibitory, non-existent and excitatory connections
#    max_min_beliefs: the maximum and minimul value of beliefs for inhibitory, non-existent and excitatory connections
#------------------------------------------------------------------------------
def calculate_belief_quality(Network,W_inferred,W_orig,l_out,whiten_flag,zero_diagonals_flag,we_know_topology):


    from scipy.cluster.vq import whiten
    
    #----------------------------Polish the Weights----------------------------
    W_inferred_our_tot = copy.deepcopy(W_inferred)
    W_inferred_our_tot = W_inferred_our_tot/float(abs(W_inferred_our_tot).max()+0.00001)
    n,m = W_inferred_our_tot.shape
    if zero_diagonals_flag:
        for i in range(0,min(n,m)):
            W_inferred_our_tot[i,i] = 0

    if whiten_flag:
        W_inferred_our_tot = W_inferred_our_tot + np.random.rand(n,m)/100000
        W_inferred_our_tot = whiten(W_inferred_our_tot)
    #--------------------------------------------------------------------------
    
    #---------Calculate Beleif Qualities for Topology Aware Algorithm----------
    if we_know_topology.lower() == 'y':
        W_inferred_our_tot = W_inferred_our_tot[:,0:m]
        W_e = np.ma.masked_array(W_inferred_our_tot,mask= (W_orig<=0).astype(int))
        mean_exc = W_e.mean(axis = 0).data
        std_exc = W_e.std(axis = 0).data
                
        W_i = np.ma.masked_array(W_inferred_our_tot,mask= (W_orig>=0).astype(int))
        mean_inh = W_i.mean(axis = 0).data
        std_inh = W_i.std(axis = 0).data
                    
        W_v = np.ma.masked_array(W_inferred_our_tot,mask= (W_orig!=0).astype(int))
        mean_void = W_v.mean(axis = 0).data
        std_void = W_v.std(axis = 0).data
        
        std_void_r = 0
        mean_void_r = 0
    #--------------------------------------------------------------------------
    
    
    #--------Calculate Beleif Qualities for Topology Unaware Algorithm---------
    else:
        ind_this_layer = 0
        n_o = ind_this_layer + Network.n_exc_array[l_out] + Network.n_inh_array[l_out]
        for l in range (0,l_out):
            ind_this_layer = ind_this_layer + Network.n_exc_array[l] + Network.n_inh_array[l]
        
        W_e = np.ma.masked_array(W_inferred_our_tot[:,ind_this_layer:ind_this_layer + n_o],mask= (W_orig[:,ind_this_layer:ind_this_layer + n_o]<=0).astype(int))        
        mean_exc = W_e.mean(axis = 0).data
        std_exc = W_e.std(axis = 0).data
                
        W_i = np.ma.masked_array(W_inferred_our_tot[:,ind_this_layer:ind_this_layer + n_o],mask= (W_orig[:,ind_this_layer:ind_this_layer + n_o]>=0).astype(int))
        mean_inh = W_i.mean(axis = 0).data
        std_inh = W_i.std(axis = 0).data
                
        W_v = np.ma.masked_array(W_inferred_our_tot[:,ind_this_layer:ind_this_layer + n_o],mask= (W_orig[:,ind_this_layer:ind_this_layer + n_o]!=0).astype(int))
        mean_void = W_v.mean(axis = 0).data
        std_void = W_v.std(axis = 0).data
        
        if Network.no_layers > 1:
            #.......Recurrent Connections in the Post-Syanptic Layer.......
            W_v_r = np.ma.masked_array(W_inferred_our_tot[ind_this_layer:ind_this_layer + n_o,ind_this_layer:ind_this_layer + n_o],mask= (W_orig[ind_this_layer:ind_this_layer + n_o,ind_this_layer:ind_this_layer + n_o]!=0).astype(int))
            mean_void_r = W_v_r.mean(axis = 0).data
            std_void_r = W_v_r.std(axis = 0).data
            #..............................................................                
        else:
            std_void_r = 0
            mean_void_r = 0
    #--------------------------------------------------------------------------
    
    means_vector = [mean_exc,mean_inh,mean_void,mean_void_r]
    std_vector = [std_exc,std_inh,std_void,std_void_r]
    
    return means_vector,std_vector
#==============================================================================
#==============================================================================


#==============================================================================
#=============================save_plot_results===============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function saves the plots corresponding to different measurment criteria
# to correspoding files (for future access, integration intolatex plots, etc.)

# INPUT:
#    T_range: the range of recording time durations considered
#    mean_exc: average value of algorithm's beliefs about "incoming" excitatory connections to this layer
#    std_exc: standard devation of algorithm's beliefs about "incoming" excitatory connections to this layer
#    mean_inh: average value of algorithm's beliefs about "incoming" inhibitory connections to this layer
#    std_inh: standard devation of algorithm's beliefs about "incoming" inhibitory connections to this layer
#    mean_void: average value of algorithm's beliefs about "incoming" void (non-existent) connections to this layer
#    std_void: standard devation of algorithm's beliefs about "incoming" void (non-existent) connections to this layer
#    mean_void_r: average value of algorithm's beliefs about "incoming" recurrent void (non-existent) connections from the same layer (if relevant)
#    std_void_r: standard devation of algorithm's beliefs about "incoming" recurrent void (non-existent) connections from the same layer (if relevant)
#    file_name_base_results: the base (usually address to the folder) where the results should be saved
#    file_name_ending: the filename endings
#    in_recurrent_flag: if 1, the code saves the results corresponding to incoming "void recurrent" links
#    W_inferred_our_tot: the inferred graph
#    W: the actual graph

# OUTPUT:
#    None
#------------------------------------------------------------------------------
def save_plot_results(T_range,mean_exc,std_exc,mean_inh,std_inh,mean_void,std_void,mean_void_r,std_void_r,file_name_base_results,file_name_ending,in_recurrent_flag,W_inferred_our_tot,W):
    
    #------------------------------Store Belief Qualities------------------------------------
    temp = np.vstack([np.array(T_range).T,(mean_exc).T,std_exc.T])
    file_name = file_name_base_results + "/Plot_Results/Mean_var_exc_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')
    
    temp = np.vstack([np.array(T_range).T,(mean_inh).T,std_inh.T])
    file_name = file_name_base_results + "/Plot_Results/Mean_var_inh_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%f',delimiter='\t',newline='\n')
    
    temp = np.vstack([np.array(T_range).T,(mean_void).T,std_void.T])
    file_name = file_name_base_results + "/Plot_Results/Mean_var_void_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%f',delimiter='\t',newline='\n')
    
    temp = np.vstack([np.array(T_range).T,(mean_exc-mean_void).T,(std_exc+std_void).T])
    file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%f',delimiter='\t',newline='\n')
    
    temp = np.vstack([np.array(T_range).T,(mean_void-mean_inh).T,(std_void+std_inh).T])
    file_name = file_name_base_results + "/Plot_Results/Gap_mean_void_inh_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')

    
    
    if in_recurrent_flag:
        temp = np.vstack([np.array(T_range).T,(mean_void_r).T,std_void_r.T])
        file_name = file_name_base_results + "/Plot_Results/Mean_var_void_recurr%s.txt" %file_name_ending
        np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')

        temp = np.vstack([np.array(T_range).T,(mean_exc-mean_void_r).T,(std_exc+std_void_r).T])
        file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_recurr%s.txt" %file_name_ending
        np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')    
    #-----------------------------------------------------------------------------------------------
    
    #-------------------------------Store the Results for Scatter Plots-----------------------------
    file_name = file_name_base_results + "/Plot_Results/Scatter_Beliefs_%s.txt" %file_name_ending
    ww = W_inferred_our_tot.ravel()
    if (len(ww) == len(np.sign(W).ravel())):
        ww = np.vstack([np.sign(W).ravel(),W_inferred_our_tot.ravel()])
        np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')
    #-----------------------------------------------------------------------------------------------
    
    #---------------------------------Save Results for Effect Sparsity------------------------------
    B_exc_void = mean_exc-mean_void
    B_void_inh = mean_void - mean_inh
        
    B_tot_ind = [0,1]
    B_tott = [B_void_inh[1],B_exc_void[1]]
    
    file_name = file_name_base_results + "/Plot_Results/Gap_final_e_v_i_%s.txt" %file_name_ending
    ww = np.vstack([B_tot_ind,B_tott])
    np.savetxt(file_name,ww.T,'%f',delimiter='\t',newline='\n')
    #-----------------------------------------------------------------------------------------------
        
    return 1
#==============================================================================
#==============================================================================


#==============================================================================
#================================save_web_demo=================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function saves the plots in a format that is intreprettable by the
# "Vis.js" javascript plugin which displays the resulting graph on the web.

# INPUT:
#    W: the actual graph
#    W_inferred_our: the inferred graph
#    file_name_base_results: the base (usually address to the folder) where the results should be saved
#    file_name_ending: the filename endings

# OUTPUT:
#    None
#------------------------------------------------------------------------------

def save_web_demo(W,W_inferred_our,file_name_base_results,file_name_ending):

    #--------------------Store the Actual Adjacency Matrix---------------------
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
    #--------------------------------------------------------------------------
    
    #------------------Store the Inferred Association Matrix-------------------    
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
    #--------------------------------------------------------------------------
    
    return 1
#==============================================================================
#==============================================================================


#==============================================================================
#=============================save_precision_recall_results===============================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function saves the plots corresponding precision and recall for different
# connection types

# INPUT:
#    T_range: the range of recording time durations considered
#    adj_fact_exc: this is the adjustment factor for clustering algorithms
#    adj_fact_inh: this is the adjustment factor for clustering algorithms
#    ternary_mode: '4' for clustering-based approach (using K-Means)
#                  '2' for sorting-based approach
#                  '7' for the conservative approach of only assigning those edges that we are sure about (far from mean values)
#    Prec_exc: a dictionary containing the average value of precisions for excitatory connections in each instance of the graphs ensemble
#    Prec_inh: a dictionary containing the average value of precisions for inhibitory connections in each instance of the graphs ensemble
#    Prec_void: a dictionary containing the average value of precisions for "void" connections in each instance of the graphs ensemble
#    Rec_exc: a dictionary containing the average value of recall for excitatory connections in each instance of the graphs ensemble
#    Rec_inh: a dictionary containing the average value of recall for inhibitory connections in each instance of the graphs ensemble
#    Rec_void: a dictionary containing the average value of recall for "void" connections in each instance of the graphs ensemble
#    std_Prec_exc: a dictionary containing the standard deviation of precisions for excitatory connections in each instance of the graphs ensemble
#    std_Prec_inh: a dictionary containing the standard deviation of precisions for inhibitory connections in each instance of the graphs ensemble
#    std_Prec_void: a dictionary containing the standard deviation of precisions for "void" connections in each instance of the graphs ensemble
#    std_Rec_exc: a dictionary containing the standard deviation of recall for excitatory connections in each instance of the graphs ensemble
#    std_Rec_inh: a dictionary containing the standard deviation of recall for inhibitory connections in each instance of the graphs ensemble
#    std_Rec_void: a dictionary containing the standard deviation of recall for "void" connections in each instance of the graphs ensemble

# OUTPUT:
#    None
#------------------------------------------------------------------------------
def save_precision_recall_results(T_range,file_name_base_result,file_name_ending_in,adj_fact_exc,adj_fact_inh,ternary_mode,Prec_exc,std_Prec_exc,Prec_inh,std_Prec_inh,Prec_void,std_Prec_void,Rec_exc,std_Rec_exc,Rec_inh,std_Rec_inh,Rec_void,std_Rec_void):
    
    #------------------------Create the File Ending----------------------------
    file_name_ending = file_name_ending_in + "_%s" %str(adj_fact_exc)
    file_name_ending = file_name_ending +"_%s" %str(adj_fact_inh)
    file_name_ending = file_name_ending + "_B_%s" %str(ternary_mode)
    file_name_ending = 'Effect_T_' + file_name_ending
    file_name_base_plot = file_name_base_result + '/Plot_Results'
    #---------------------------------------------------------------------------
        
    #--------------------------Store Precision and Recall-----------------------    
    temp = np.vstack([T_range,Prec_exc,std_Prec_exc])
    file_name = file_name_base_plot + "/Prec_exc_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

    temp = np.vstack([T_range,Prec_inh,std_Prec_inh])
    file_name = file_name_base_plot + "/Prec_inh_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')
    
    temp = np.vstack([T_range,Prec_void,std_Prec_void])
    file_name = file_name_base_plot + "/Prec_void_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

    temp = np.vstack([T_range,Rec_exc,std_Rec_exc])
    file_name = file_name_base_plot + "/Reca_exc_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

    temp = np.vstack([T_range,Rec_inh,std_Rec_inh])
    file_name = file_name_base_plot + "/Reca_inh_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

    temp = np.vstack([T_range,Rec_void,std_Rec_void])
    file_name = file_name_base_plot + "/Reca_void_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

    temp = np.vstack([Rec_exc,Prec_exc,std_Rec_exc,std_Prec_exc])
    file_name = file_name_base_plot + "/Reca_vs_Prec_exc_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')

    temp = np.vstack([Rec_inh,Prec_inh,std_Rec_inh,std_Prec_inh])
    file_name = file_name_base_plot + "/Reca_vs_Prec_inh_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')
    
    temp = np.vstack([Rec_void,Prec_void,std_Rec_void,std_Prec_void])
    file_name = file_name_base_plot + "/Reca_vs_Prec_void_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')
    #---------------------------------------------------------------------------
    
        
    return 1
#==============================================================================
#==============================================================================


#==============================================================================
#================================save_web_demo=================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function saves the plots in a format that is intreprettable by the
# "Vis.js" javascript plugin which displays the resulting graph on the web.

# INPUT:
#    Network: the object that contains the informatin about the connectivity pattern of each layer
#    we_know_topology: if 'Y', the inference has been topology-aware
#    T_range: the range of recording times we plot our variables over

# OUTPUT:
#    mean_exc: a dictionary containing the average value of beliefs for excitatory connections in each layer
#    mean_inh: a dictionary containing the average value of beliefs for inhibitory connections in each layer
#    mean_void: a dictionary containing the average value of beliefs for "void" connections in each layer
#    mean_void_r: a dictionary containing the average value of beliefs for "void recurrent" connections in each layer (if relevant)
#    std_exc: a dictionary containing the standard deviation of beliefs for excitatory connections in each layer
#    std_inh: a dictionary containing the standard deviation of beliefs for inhibitory connections in each layer
#    std_void: a dictionary containing the standard deviation of beliefs for "void" connections in each layer
#    std_void_r: a dictionary containing the standard deviation of beliefs for "void recurrent" connections in each layer (if relevant)
#    Prec_exc: a dictionary containing the average value of precisions for excitatory connections in each instance of the graphs ensemble
#    Prec_inh: a dictionary containing the average value of precisions for inhibitory connections in each instance of the graphs ensemble
#    Prec_void: a dictionary containing the average value of precisions for "void" connections in each instance of the graphs ensemble
#    Rec_exc: a dictionary containing the average value of recall for excitatory connections in each instance of the graphs ensemble
#    Rec_inh: a dictionary containing the average value of recall for inhibitory connections in each instance of the graphs ensemble
#    Rec_void: a dictionary containing the average value of recall for "void" connections in each instance of the graphs ensemble
#    std_Prec_exc: a dictionary containing the standard deviation of precisions for excitatory connections in each instance of the graphs ensemble
#    std_Prec_inh: a dictionary containing the standard deviation of precisions for inhibitory connections in each instance of the graphs ensemble
#    std_Prec_void: a dictionary containing the standard deviation of precisions for "void" connections in each instance of the graphs ensemble
#    std_Rec_exc: a dictionary containing the standard deviation of recall for excitatory connections in each instance of the graphs ensemble
#    std_Rec_inh: a dictionary containing the standard deviation of recall for inhibitory connections in each instance of the graphs ensemble
#    std_Rec_void: a dictionary containing the standard deviation of recall for "void" connections in each instance of the graphs ensemble
#------------------------------------------------------------------------------

def initialize_plotting_variables(Network,we_know_topology,T_range,ensemble_size,ensemble_count_init):
    mean_exc = {};mean_inh = {};mean_void = {};std_void_r = {};
    std_exc = {};std_inh = {};std_void = {};mean_void_r = {}    
    Prec_exc = {};Prec_inh = {}; Prec_void = {};
    Rec_exc = {};Rec_inh = {}; Rec_void = {};
    std_Prec_exc = {};std_Prec_inh = {}; std_Prec_void = {};
    std_Rec_exc = {};std_Rec_inh = {}; std_Rec_void = {};
    
    for l_out in range(0,Network.no_layers):    
        n_layer = Network.n_exc_array[l_out]+Network.n_inh_array[l_out]
        if we_know_topology.lower() == 'n':
            ind = str(l_out)
            std_exc[ind] = np.zeros([len(T_range),n_layer])             # The standard deviation of beliefs for excitatory connections
            std_inh[ind] = np.zeros([len(T_range),n_layer])             # The standard deviation of beliefs for inhibitory connections
            std_void[ind] = np.zeros([len(T_range),n_layer])            # The standard deviation of beliefs for "void" connections
            mean_exc[ind] = np.zeros([len(T_range),n_layer])            # The average value of beliefs for excitatory connections
            mean_inh[ind] = np.zeros([len(T_range),n_layer])            # The average value of beliefs for inhibitory connections
            mean_void[ind] = np.zeros([len(T_range),n_layer])           # The average value of beliefs for "void" connections
            std_void_r[ind] = np.zeros([len(T_range),n_layer])
            mean_void_r[ind] = np.zeros([len(T_range),n_layer])
        
            Prec_exc[ind] = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                        # Detailed precision of the algorithm (per ensemble) for excitatory connections
            Prec_inh[ind] = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                        # Detailed precision of the algorithm (per ensemble) for inhibitory connections
            Prec_void[ind]  = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                      # Detailed precision of the algorithm (per ensemble) for "void" connections
        
            Rec_exc[ind] = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                         # Detailed recall of the algorithm for (per ensemble) excitatory connections
            Rec_inh[ind] = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                         # Detailed recall of the algorithm for (per ensemble) inhibitory connections
            Rec_void[ind]  = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                       # Detailed recall of the algorithm for (per ensemble) "void" connections
        
            std_Prec_exc[ind] = np.zeros([len(T_range)])                                      # Standard Deviation of precision of the algorithm for excitatory connections
            std_Prec_inh[ind] = np.zeros([len(T_range)])                                      # Standard Deviation of precision of the algorithm for inhibitory connections
            std_Prec_void[ind]  = np.zeros([len(T_range)])                                    # Standard Deviation of precision of the algorithm for "void" connections
        
            std_Rec_exc[ind] = np.zeros([len(T_range)])                                       # Standard Deviation of recall of the algorithm for excitatory connections
            std_Rec_inh[ind] = np.zeros([len(T_range)])                                       # Standard Deviation of recall of the algorithm for excitatory connections
            std_Rec_void[ind]  = np.zeros([len(T_range)])                                     # Standard Deviation of recall of the algorithm for excitatory connections

        else:
            for l_in in range (0,l_out + 1):
                ind = str(l_in) + str(l_out)
                std_exc[ind] = np.zeros([len(T_range),n_layer])
                std_inh[ind] = np.zeros([len(T_range),n_layer])
                std_void[ind] = np.zeros([len(T_range),n_layer])
                mean_exc[ind] = np.zeros([len(T_range),n_layer])
                mean_inh[ind] = np.zeros([len(T_range),n_layer])
                mean_void[ind] = np.zeros([len(T_range),n_layer])
                std_void_r[ind] = np.zeros([len(T_range),n_layer])
                mean_void_r[ind] = np.zeros([len(T_range),n_layer])
                
                Prec_exc[ind] = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                        # Detailed precision of the algorithm (per ensemble) for excitatory connections
                Prec_inh[ind] = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                        # Detailed precision of the algorithm (per ensemble) for inhibitory connections
                Prec_void[ind]  = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                      # Detailed precision of the algorithm (per ensemble) for "void" connections
            
                Rec_exc[ind] = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                         # Detailed recall of the algorithm for (per ensemble) excitatory connections
                Rec_inh[ind] = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                         # Detailed recall of the algorithm for (per ensemble) inhibitory connections
                Rec_void[ind]  = np.zeros([len(T_range),ensemble_size-ensemble_count_init])                       # Detailed recall of the algorithm for (per ensemble) "void" connections
                
                std_Prec_exc[ind] = np.zeros([len(T_range)])                                      # Standard Deviation of precision of the algorithm for excitatory connections
                std_Prec_inh[ind] = np.zeros([len(T_range)])                                      # Standard Deviation of precision of the algorithm for inhibitory connections
                std_Prec_void[ind]  = np.zeros([len(T_range)])                                    # Standard Deviation of precision of the algorithm for "void" connections
            
                std_Rec_exc[ind] = np.zeros([len(T_range)])                                       # Standard Deviation of recall of the algorithm for excitatory connections
                std_Rec_inh[ind] = np.zeros([len(T_range)])                                       # Standard Deviation of recall of the algorithm for excitatory connections
                std_Rec_void[ind]  = np.zeros([len(T_range)])                                     # Standard Deviation of recall of the algorithm for excitatory connections
                
    
    return mean_exc,mean_inh,mean_void,std_void_r,std_exc,std_inh,std_void,mean_void_r,Prec_exc,Prec_inh,Prec_void,Rec_exc,Rec_inh,Rec_void,std_Prec_exc,std_Prec_inh,std_Prec_void,std_Rec_exc,std_Rec_inh,std_Rec_void
#------------------------------------------------------------------------------

#==============================================================================
#==============================================================================


#==============================================================================
#==============================export_to_plotly================================
#==============================================================================
#-------------------------------Descriptions-----------------------------------
# This function saves the plots in a format that is intreprettable by the
# "Vis.js" javascript plugin which displays the resulting graph on the web.

# INPUT:
#    W: the actual graph
#    W_inferred_our: the inferred graph
#    file_name_base_results: the base (usually address to the folder) where the results should be saved
#    file_name_ending: the filename endings

# OUTPUT:
#    None
#------------------------------------------------------------------------------

def export_to_plotly(x_array,y_array,no_plots,plot_legends,plot_type,plot_colors,x_label,y_label,plot_title,error_array=[],error_bar_colors = []):
    
    if len(error_array):
        error_flag = 1
    else:
        error_flag = 0
    
    traces = []
    
    for i in range (0,no_plots):
        X = x_array[i,:]
        Y = y_array[i,:]
        if error_flag:
            e = error_array[i,:]
    
            if plot_type == 'line':
                trace = Scatter(
                    x=X,
                    y=Y,
                    name=plot_legends[i],
                    error_y=ErrorY(
                        type='data',
                        array=e,
                        visible=True,
                        color=error_bar_colors[i],
                        ),
                    marker=Marker(
                       color=plot_colors[i]
                    ),
                    )
            else:
                trace = Bar(
                    x=X,
                    y=Y,
                    marker=Marker(
                       color=plot_colors[i]
                    ),
                    name=plot_legends[i],
                    error_y=ErrorY(
                        type='data',
                        array=e,
                        visible=True,
                        color=error_bar_colors[i],
                        )
                    )
        else:
            
            if plot_type == 'line':
                trace = Scatter(
                    x=X,
                    y=Y,
                    name=plot_legends[i],
                    marker=Marker(
                       color=plot_colors[i],
                    ),
                    )
            else:
                trace = Bar(
                    x=X,
                    y=Y,
                    name=plot_legends[i],
                    marker=Marker(
                       color=plot_colors[i]
                    ),
                    )
            
        
        traces.append(trace)
    data = Data(traces)
    
    
    if plot_type == 'bar':
        layout = Layout(
            title=plot_title,
            barmode='group',
            xaxis=XAxis(
                title=x_label,                
            ),
            yaxis=YAxis(
                title=y_label,                
            ),
        )
    else:
        layout = Layout(
            title=plot_title, 
            xaxis=XAxis(
                title=x_label,                
            ),
            yaxis=YAxis(
                title=y_label,                
            ),
        )
        
    
    fig = Figure(data=data, layout=layout)
    if plot_type == 'bar':        
        if error_flag:
            plot_url = pltly.plot(fig, filename='error-bar-barr')
        else:
            plot_url = pltly.plot(fig, filename='basic-bar')
    else:
        if error_flag:
            plot_url = pltly.plot(fig, filename='basic-error-bar')
        else:
            plot_url = pltly.plot(fig, filename='basic-line')
    
    
    return plot_url