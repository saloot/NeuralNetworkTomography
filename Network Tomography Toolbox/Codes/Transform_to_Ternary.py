#=======================IMPORT THE NECESSARY LIBRARIES=========================
#from brian import *
import time
import numpy as np
import sys,getopt,os
from scipy.cluster.vq import kmeans,whiten,kmeans2,vq

#from CommonFunctions.auxiliary_functions import combine_weight_matrix,generate_file_name
from CommonFunctions.auxiliary_functions_digitize import beliefs_to_ternary,parse_commands_ternary_algo
#from CommonFunctions.Neurons_and_Networks import *

os.system('clear')                                              # Clear the commandline window
#==============================================================================


#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:K:C:Y:U:Z:o:N:H:j:c:")

file_name_ending_list,file_name_base_results,ternary_mode,n,no_hidden_neurons,adj_fact_exc,adj_fact_inh = parse_commands_ternary_algo(input_opts)

m = n
#==============================================================================

#================================INITIALIZATIONS===============================

#-----------------------Set Simulation Variables------------------------
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

#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
W_inferred = np.zeros([n-no_hidden_neurons,1])
W_infer = np.zeros([len(file_name_ending_list),n-no_hidden_neurons])

#------------------------Read the Inferred Weights-------------------------
itr_i = 0
for file_name_ending in file_name_ending_list:
    file_name = "Inferred_Graphs/" + file_name_ending
    file_name = file_name_base_results + '/' + file_name        
    W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    W_infer[itr_i,:] = W_read[0:n-no_hidden_neurons]

    if itr_i > 1:
        W_inferred[0:min(m-no_hidden_neurons,len(W_read)),0] = W_infer[0:itr_i,:].mean(axis = 0)
    else:
        W_inferred[0:min(m-no_hidden_neurons,len(W_read)),0] = W_infer[itr_i-1,:]
    
    W_inferred_s = W_inferred#[:,neuron_range]
    #W_inferred_s = W_inferred_s[:-1]
    #--------------------------------------------------------------------------

    #-----------------Calculate the Binary Matrix From Beliefs-----------------
    W_binary,centroids = beliefs_to_ternary(ternary_mode,W_inferred_s,params,dale_law_flag)                    
    #--------------------------------------------------------------------------
    
    #--------------------------Store the Binary Matrices-----------------------
    file_name_ending2 = file_name_ending.replace('.txt','') + "_%s" %str(adj_fact_exc)
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
                    
#======================================================================================
