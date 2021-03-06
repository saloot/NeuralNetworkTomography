#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
import time
import numpy as np
import sys,getopt,os
from CommonFunctions.auxiliary_functions import remap_connections
import pdb 
#from CommonFunctions.auxiliary_functions import combine_weight_matrix,generate_file_name
from CommonFunctions.auxiliary_functions_digitize import beliefs_to_ternary,parse_commands_ternary_algo
from CommonFunctions.auxiliary_functions import enforce_structural_connections
#from CommonFunctions.Neurons_and_Networks import *
try:
    import matplotlib.pyplot as plt
except:
    pass
os.system('clear')                                              # Clear the commandline window
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:K:C:Y:U:Z:o:N:H:j:c:f:n:")

file_name_ending_list,file_name_base_results,ternary_mode,n,n_ind,no_hidden_neurons,no_structural_connections,adj_fact_exc,adj_fact_inh = parse_commands_ternary_algo(input_opts)

m = n
#==============================================================================

#================================INITIALIZATIONS===============================

#-----------------------Set Simulation Variables------------------------
dale_law_flag = 0   # If 1, the ternarification algorithm returns a matrix in which the non-zero entries in a row (i.e. outgoing neural connections) have the same sign
#------------------------------------------------------------------------------

#---------------------Initialize Ternarification Parameters--------------------

#.............................Sorting-Based Approach...........................
if ternary_mode == 2:    
    params = [adj_fact_exc,adj_fact_inh]                     # Parameters will be set later
#..............................................................................

#..........................Clustering-based Approach...........................
if ternary_mode == 4:
    params = [adj_fact_exc,adj_fact_inh,[]]
#..............................................................................

#------------------------------------------------------------------------------

#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
n_neuorns = n
if no_hidden_neurons:
    n_neuorns -= (no_hidden_neurons+1)
#W_infer = np.zeros([len(file_name_ending_list),n_neuorns])
#W_inferred = np.zeros([n,1])
#W_infer = np.zeros([len(file_name_ending_list),n])

#------------------------Read the Inferred Weights-------------------------
itr_i = 0
for file_name_ending in file_name_ending_list:
    file_name = "Inferred_Graphs/" + file_name_ending
    file_name = file_name_base_results + '/' + file_name        
    W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #W_infer[itr_i,:] = W_read[0:n_neuorns]
    W_inferred = W_read[:-1]#[0:n_neuorns]
    #W_infer[itr_i,:] = W_read[0:n]
    #W_inferred[0:m,0] = W_infer[itr_i,:]
    #--------------------------------------------------------------------------

    #-----------Reconstruct Based on Available Structural Information----------
    if no_structural_connections:
        file_name_ending_mod = file_name_ending.replace('W_Pll_','')
        file_name_hidden = "../Results/Inferred_Graphs/Hidden_or_Structured_Neurons_" + file_name_ending_mod
        structural_neurons = np.genfromtxt(file_name_hidden, dtype=None, delimiter='\t')
        structural_neurons = np.hstack([structural_neurons,n_ind])
        
        #W_inferred = W_inferred - W_inferred.mean()
        #W_inferred = enforce_structural_connections(W_inferred,structural_neurons)
        #W_inferred /= W_inferred.max()
        
    else:
        structural_neurons = [n_ind]
    #--------------------------------------------------------------------------


    #-----------------Calculate the Binary Matrix From Beliefs-----------------
    W_binary,centroids = beliefs_to_ternary(ternary_mode,W_inferred,params,dale_law_flag)
    if len(structural_neurons):
        #W_binary = enforce_structural_connections(W_binary,structural_neurons)
        W_binary = remap_connections(W_binary,structural_neurons,n)
    #--------------------------------------------------------------------------
    
    #--------------------------Store the Binary Matrices-----------------------
    file_name_ending2 = file_name_ending.replace('.txt','') + "_%s" %str(adj_fact_exc)
    file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
    file_name_ending2 = file_name_ending2 + "_B_%s" %str(ternary_mode)
    #try:
    #    ind = file_name_ending2.index('_ID_')
    #    aa = file_name_ending2[ind:ind+10]
    #    file_name_ending2 = file_name_ending2.replace(aa,'')
    #except:
    #    pass
             
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
                    
#======================================================================================
