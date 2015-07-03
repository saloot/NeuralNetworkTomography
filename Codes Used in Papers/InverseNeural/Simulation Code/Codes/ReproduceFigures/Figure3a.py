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
from CommonFunctions.auxiliary_functions_digitize import parse_commands_ternary_algo
from CommonFunctions.auxiliary_functions import generate_file_name
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

Network.read_weights(0,file_name_base_data)

#==============================================================================

#=============================READ THE RESULTS=================================

#------------------------Cross Correlogram Approach----------------------------
T = T_range[0]

file_name_ending = generate_file_name(Network.file_name_ending,4,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')        
file_name_ending = file_name_ending + '_1' 
file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_%s.txt" %file_name_ending

gap_exc_void_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')
gap_exc_void_cc = gap_exc_void_cc[:,1]

file_name = file_name_base_results + "/Plot_Results/Gap_mean_void_inh_%s.txt" %file_name_ending
gap_void_inh_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')

T_range_cc = gap_void_inh_cc[:,0]
gap_void_inh_cc = gap_void_inh_cc[:,1]
#------------------------------------------------------------------------------


#--------------------------------GLM Approach----------------------------------
T = T_range[1]

file_name_ending = generate_file_name(Network.file_name_ending,8,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')        
file_name_ending = file_name_ending + '_1' 
file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_%s.txt" %file_name_ending

gap_exc_void_glm = np.genfromtxt(file_name, dtype='float', delimiter='\t')
gap_exc_void_glm = gap_exc_void_glm[:,1]

file_name = file_name_base_results + "/Plot_Results/Gap_mean_void_inh_%s.txt" %file_name_ending
gap_void_inh_glm = np.genfromtxt(file_name, dtype='float', delimiter='\t')

T_range_glm = gap_void_inh_glm[:,0]
gap_void_inh_glm = gap_void_inh_glm[:,1]
#------------------------------------------------------------------------------

#-------------------------------NeuInf Approach--------------------------------
T = T_range[2]

file_name_ending = generate_file_name(Network.file_name_ending,3,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')        
file_name_ending = file_name_ending + '_1' 
file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_%s.txt" %file_name_ending

gap_exc_void_neuinf = np.genfromtxt(file_name, dtype='float', delimiter='\t')
gap_exc_void_neuinf = gap_exc_void_neuinf[:,1]

file_name = file_name_base_results + "/Plot_Results/Gap_mean_void_inh_%s.txt" %file_name_ending
gap_void_inh_neuinf = np.genfromtxt(file_name, dtype='float', delimiter='\t')

T_range_neuinf = gap_void_inh_neuinf[:,0]
gap_void_inh_neuinf = gap_void_inh_neuinf[:,1]
#------------------------------------------------------------------------------

#-------------------------------Location Known---------------------------------
T = T_range[3]

temp = Network.file_name_ending + '_l_0_to_1'
file_name_ending = generate_file_name(temp,3,'Y','A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')

file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_%s.txt" %file_name_ending

gap_exc_void_neuinf_loc_y = np.genfromtxt(file_name, dtype='float', delimiter='\t')
gap_exc_void_neuinf_loc_y = gap_exc_void_neuinf_loc_y[:,1]

file_name = file_name_base_results + "/Plot_Results/Gap_mean_void_inh_%s.txt" %file_name_ending
gap_void_inh_neuinf_loc_y = np.genfromtxt(file_name, dtype='float', delimiter='\t')

gap_void_inh_neuinf_loc_y = gap_void_inh_neuinf_loc_y[:,1]
#------------------------------------------------------------------------------

#==============================================================================



#=================================PLOT THE RESULTS=============================
bar_width = 0.35
fig, axs = plt.subplots(nrows=2, ncols=1)
ax = axs[0]
ax.bar(T_range_glm,gap_exc_void_glm,bar_width,color='b',label='GLM');
ax.bar(T_range_neuinf + bar_width,gap_exc_void_cc,bar_width,color='r',label='Cross Correlogram');
ax.bar(T_range_neuinf + 2*bar_width,gap_void_inh_neuinf,bar_width,color='orange',label='NeuInf');
ax.bar(T_range_neuinf + 3*bar_width,gap_void_inh_neuinf_loc_y,bar_width,color='white',edgecolor='orange',hatch="//",label='NeuInf, Top. Aware');
ax.set_title('Inhibitory-Void')
ax.grid()
ax.legend(loc='upper left')

ax = axs[1]
ax.bar(T_range_glm,gap_void_inh_glm,bar_width,color='b',label='GLM');
ax.bar(T_range_neuinf + bar_width,gap_void_inh_cc,bar_width,color='r',label='Cross Correlogram');
ax.bar(T_range_neuinf + 2*bar_width,gap_exc_void_neuinf,bar_width,color='orange',label='NeuInf');
ax.bar(T_range_neuinf + 3*bar_width,gap_exc_void_neuinf_loc_y,bar_width,color='white',edgecolor='orange',hatch="//",label='NeuInf, Top. Aware');
ax.set_title('Excitatory-Void')
ax.grid()
ax.legend(loc='upper left')


#plt.title('Belief gaps')
plt.xlabel('T(s)', fontsize=16)

plt.show();
#==============================================================================

