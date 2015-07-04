#=======================IMPORT THE NECESSARY LIBRARIES=========================
import numpy as np
import sys,getopt,os
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
import matplotlib.mlab as mlab

try: 
    import plotly.plotly as pltly
    from plotly.graph_objs import *
    plotly_import = 1
except:
    print 'Plotly was not found. No problem though, life goes on ;)'
    plotly_import = 0

os.chdir('../')
sys.path.append('./')


from CommonFunctions.Neurons_and_Networks import *
from CommonFunctions.default_values import *
from CommonFunctions.auxiliary_functions_digitize import parse_commands_ternary_algo,beliefs_to_ternary
from CommonFunctions.auxiliary_functions import generate_file_name,combine_weight_matrix
from CommonFunctions.auxiliary_functions_plot import export_to_plotly

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

#----------------------------Get the Actual Graph------------------------------
Network.read_weights(0,file_name_base_data)
W,DD_tot = combine_weight_matrix(Network)
#------------------------------------------------------------------------------

#--------------------------Get the Inferred Graph------------------------------
file_name_ending = generate_file_name(Network.file_name_ending,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T_max,'N')
file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
try:
    W_inferred = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    found_file_tot = 1
except:
    found_file_tot = 0
    print 'Sorry I can not find the corresponding inference file for the netwrok'
#------------------------------------------------------------------------------


#---------------------------Get the Binary Graph-------------------------------
if found_file_tot:
    n,m = W.shape
    fixed_entries = np.zeros([n,m])
    W_inferred = W_inferred[:,0:m]
    P = np.sum(abs(W)>0)/float(m*n)
    p_exc = P * (np.sum(W>0)/float(np.sum(abs(W)>0)))
    p_inh = P * (np.sum(W<0)/float(np.sum(abs(W)>0)))
    
    params=[p_exc,p_inh]
                        
    W_inferred = W_inferred/float(abs(W_inferred).max())
    W_inferred = W_inferred + np.random.rand(n,m)/100000                    # This is done to keep the whitening process screwing up all-zero columns
    W_inferred = whiten(W_inferred)
    
    for i in range(0,n):
        W_inferred[i,i] = 0
    
    W_binary,centroids = beliefs_to_ternary(2,10*W_inferred,params,0)
#------------------------------------------------------------------------------

#==============================================================================



#=================================PLOT THE RESULTS=============================
fig, axs = plt.subplots(nrows=1, ncols=3)
ax = axs[0]
ax.imshow(np.sign(W))
ax.set_title('Actual Graph')

ax = axs[1]
ax.imshow(W_inferred)
ax.set_title('STOCHASTIC NEUINF')

ax = axs[2]
ax.imshow(W_binary)
ax.set_title('Ternary STOCHASTIC NEUINF')


plt.show();
#==============================================================================

