#=======================IMPORT THE NECESSARY LIBRARIES=========================
import numpy as np
import sys,getopt,os
from time import time
import matplotlib.pyplot as plt
import copy
from scipy.cluster.vq import whiten
import matplotlib.mlab as mlab
try: 
    import plotly.plotly as pltly
    from plotly.graph_objs import *
    plotly_import = 1
except:
    print 'Plotly was not found. No problem though, life goes on ;)'
    plotly_import = 0


from Neurons_and_Networks import *
from default_values import *
from auxiliary_functions_digitize import parse_commands_ternary_algo
from CommonFunctions.auxiliary_functions import generate_file_name,combine_weight_matrix
from CommonFunctions.auxiliary_functions_plot import save_plot_results,calculate_belief_quality,save_web_demo,initialize_plotting_variables,save_precision_recall_results,export_to_plotly

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:J:K:U:Z:Y:O:f:")
frac_stimulated_neurons,T_max,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,ternary_mode,file_name_base_results,inference_method,sparsity_flag,we_know_topology,beta,alpha0,infer_itr_max,T_range,plot_flags = parse_commands_ternary_algo(input_opts)
#==============================================================================

#================================INITIALIZATIONS===============================

#--------------------------Initialize the Network------------------------------
Network = NeuralNet(None,None,None,None,None,None,None, 'command_line',input_opts,args)
#------------------------------------------------------------------------------    

#------Calculate the Range to Assess the Effect of Recording Duration------
if not T_range:
    T_step = int(T_max/6.0)
    T_range = range(T_step, T_max+1, T_step)
#T_range = [2779,5458,8137,10816]
#--------------------------------------------------------------------------


#-----------------------------Signin to Plotly---------------------------------
plotly_flag = 0                     # If '1', figures will be exported to PlotLy (https://plot.ly/) as well
if not plotly_import:               # If PlotLy was not installed, there is no need to consider its plottings
    plotly_flag = 0
    
if plotly_flag:
    usrname = raw_input("What is your PlotLy username? ")
    passwd = raw_input("What is your PlotLy API Key? ")
    if not passwd:
        passwd = 'wj370nzaqg'
    pltly.sign_in(usrname, passwd)
#------------------------------------------------------------------------------

#==============================================================================

#=============================READ THE RESULTS=================================
T = T_range[0]
file_name_ending = generate_file_name(Network.file_name_ending,4,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')        
file_name_ending = file_name_ending + '_1' 
file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_%s.txt" %file_name_ending

gap_exc_void_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')


file_name = file_name_base_results + "/Plot_Results/Gap_mean_void_inh_%s.txt" %file_name_ending
gap_void_inh_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')

#==============================================================================

