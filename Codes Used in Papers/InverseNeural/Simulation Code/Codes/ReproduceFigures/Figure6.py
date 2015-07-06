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
Network.read_weights(0,file_name_base_data)
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

#--------------------------Initialize Other Variables--------------------------
adj_fact_exc = 0.75                 # This is the adjustment factor for clustering algorithms (between 0 and infinity).
adj_fact_inh = 0.5                  # This is the adjustment factor for clustering algorithms (between 0 and infinity).
#------------------------------------------------------------------------------

#==============================================================================

#=============================READ THE RESULTS=================================

#------------------------CC Approach----------------------------
T = T_range[0]

file_name_ending = generate_file_name(Network.file_name_ending,4,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')
file_name_ending = file_name_ending + '_0'
file_name_ending = file_name_ending.replace('_T_%s'%str(T),'')
file_name_ending = file_name_ending + "_%s" %str(adj_fact_exc)
file_name_ending = file_name_ending +"_%s" %str(adj_fact_inh)
file_name_ending = file_name_ending + "_B_%s" %str(ternary_mode)
file_name_ending = 'Effect_T_' + file_name_ending
    
file_name = file_name_base_results + "/Plot_Results/Prec_exc_%s.txt" %file_name_ending
prec_exc_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')
prec_exc_cc = prec_exc_cc[:,1]

file_name = file_name_base_results + "/Plot_Results/Prec_void_%s.txt" %file_name_ending
prec_void_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')
prec_void_cc = prec_void_cc[:,1]

file_name = file_name_base_results + "/Plot_Results/Prec_inh_%s.txt" %file_name_ending
prec_inh_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')
T_range_cc = prec_inh_cc[:,0]
prec_inh_cc = prec_inh_cc[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_exc_%s.txt" %file_name_ending
rec_exc_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_exc_cc = rec_exc_cc[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_void_%s.txt" %file_name_ending
rec_void_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_void_cc = rec_void_cc[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_inh_%s.txt" %file_name_ending
rec_inh_cc = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_inh_cc = rec_inh_cc[:,1]
#------------------------------------------------------------------------------

#--------------------------------GLM Approach----------------------------------
T = T_range[1]

file_name_ending = generate_file_name(Network.file_name_ending,8,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')
file_name_ending = file_name_ending + '_0'
file_name_ending = file_name_ending.replace('_T_%s'%str(T),'')
file_name_ending = file_name_ending + "_%s" %str(adj_fact_exc)
file_name_ending = file_name_ending +"_%s" %str(adj_fact_inh)
file_name_ending = file_name_ending + "_B_%s" %str(ternary_mode)
file_name_ending = 'Effect_T_' + file_name_ending
    
file_name = file_name_base_results + "/Plot_Results/Prec_exc_%s.txt" %file_name_ending
prec_exc_glm = np.genfromtxt(file_name, dtype='float', delimiter='\t')
prec_exc_glm = prec_exc_glm[:,1]

file_name = file_name_base_results + "/Plot_Results/Prec_void_%s.txt" %file_name_ending
prec_void_glm = np.genfromtxt(file_name, dtype='float', delimiter='\t')
prec_void_glm = prec_void_glm[:,1]

file_name = file_name_base_results + "/Plot_Results/Prec_inh_%s.txt" %file_name_ending
prec_inh_glm = np.genfromtxt(file_name, dtype='float', delimiter='\t')
T_range_glm = prec_inh_glm[:,0]
prec_inh_glm = prec_inh_glm[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_exc_%s.txt" %file_name_ending
rec_exc_glm = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_exc_glm = rec_exc_glm[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_void_%s.txt" %file_name_ending
rec_void_glm = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_void_glm = rec_void_glm[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_inh_%s.txt" %file_name_ending
rec_inh_glm = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_inh_glm = rec_inh_glm[:,1]
#------------------------------------------------------------------------------

#-------------------------------NeuInf Approach--------------------------------
T = T_range[2]

file_name_ending = generate_file_name(Network.file_name_ending,3,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')
file_name_ending = file_name_ending + '_0'
file_name_ending = file_name_ending.replace('_T_%s'%str(T),'')
file_name_ending = file_name_ending + "_%s" %str(adj_fact_exc)
file_name_ending = file_name_ending +"_%s" %str(adj_fact_inh)
file_name_ending = file_name_ending + "_B_%s" %str(ternary_mode)
file_name_ending = 'Effect_T_' + file_name_ending
    
file_name = file_name_base_results + "/Plot_Results/Prec_exc_%s.txt" %file_name_ending
prec_exc_neuinf = np.genfromtxt(file_name, dtype='float', delimiter='\t')
prec_exc_neuinf = prec_exc_neuinf[:,1]

file_name = file_name_base_results + "/Plot_Results/Prec_void_%s.txt" %file_name_ending
prec_void_neuinf = np.genfromtxt(file_name, dtype='float', delimiter='\t')
prec_void_neuinf = prec_void_neuinf[:,1]

file_name = file_name_base_results + "/Plot_Results/Prec_inh_%s.txt" %file_name_ending
prec_inh_neuinf = np.genfromtxt(file_name, dtype='float', delimiter='\t')
T_range_neuinf = prec_inh_neuinf[:,0]
prec_inh_neuinf = prec_inh_neuinf[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_exc_%s.txt" %file_name_ending
rec_exc_neuinf = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_exc_neuinf = rec_exc_neuinf[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_void_%s.txt" %file_name_ending
rec_void_neuinf = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_void_neuinf = rec_void_neuinf[:,1]

file_name = file_name_base_results + "/Plot_Results/Reca_inh_%s.txt" %file_name_ending
rec_inh_neuinf = np.genfromtxt(file_name, dtype='float', delimiter='\t')
rec_inh_neuinf = rec_inh_neuinf[:,1]
#------------------------------------------------------------------------------

#==============================================================================



#=================================PLOT THE RESULTS=============================
bar_width = 0.1
fig, axs = plt.subplots(nrows=3, ncols=1)
ax = axs[0]
ax.bar(T_range_neuinf + 0*bar_width,prec_void_neuinf,bar_width,color='orange',label='NeuInf, Prec.');
ax.bar(T_range_neuinf + 1*bar_width,prec_void_glm,bar_width,color='r',label='GLM, Prec.');
ax.bar(T_range_neuinf + 2*bar_width,prec_void_cc,bar_width,color='b',label='CC, Prec.');

ax.bar(T_range_neuinf + 3*bar_width,rec_void_neuinf,bar_width,color='white',edgecolor='orange',hatch="//",label='NeuInf, Rec.');
ax.bar(T_range_neuinf + 4*bar_width,rec_void_glm,bar_width,color='white',edgecolor='red',hatch="//",label='GLM, Rec.');
ax.bar(T_range_neuinf + 5*bar_width,rec_void_cc,bar_width,color='white',edgecolor='blue',hatch="//",label='CC, Rec.');
ax.set_title('Void')
ax.grid()
ax.legend(loc='upper left',ncol=2,prop={'size':10})

ax = axs[1]
ax.bar(T_range_neuinf + 0*bar_width,prec_inh_neuinf,bar_width,color='orange',label='NeuInf, Prec.');
ax.bar(T_range_neuinf + 1*bar_width,prec_inh_glm,bar_width,color='r',label='GLM, Prec.');
ax.bar(T_range_neuinf + 2*bar_width,prec_inh_cc,bar_width,color='b',label='CC, Prec.');

ax.bar(T_range_neuinf + 3*bar_width,rec_inh_neuinf,bar_width,color='white',edgecolor='orange',hatch="//",label='NeuInf, Rec.');
ax.bar(T_range_neuinf + 4*bar_width,rec_inh_glm,bar_width,color='white',edgecolor='red',hatch="//",label='GLM, Rec.');
ax.bar(T_range_neuinf + 5*bar_width,rec_inh_cc,bar_width,color='white',edgecolor='blue',hatch="//",label='CC, Rec.');
ax.set_title('Inhibitory')
ax.grid()
ax.legend(loc='upper left',ncol=2,prop={'size':10})

ax = axs[2]
ax.bar(T_range_neuinf + 0*bar_width,prec_exc_neuinf,bar_width,color='orange',label='NeuInf, Prec.');
ax.bar(T_range_neuinf + 1*bar_width,prec_exc_glm,bar_width,color='r',label='GLM, Prec.');
ax.bar(T_range_neuinf + 2*bar_width,prec_exc_cc,bar_width,color='b',label='CC, Prec.');

ax.bar(T_range_neuinf + 3*bar_width,rec_exc_neuinf,bar_width,color='white',edgecolor='orange',hatch="//",label='NeuInf, Rec.');
ax.bar(T_range_neuinf + 4*bar_width,rec_exc_glm,bar_width,color='white',edgecolor='red',hatch="//",label='GLM, Rec.');
ax.bar(T_range_neuinf + 5*bar_width,rec_exc_cc,bar_width,color='white',edgecolor='blue',hatch="//",label='CC, Rec.');
ax.set_title('Excitatory')
ax.grid()
ax.legend(loc='upper left',ncol=2,prop={'size':10})

#plt.title('Belief gaps')
plt.xlabel('T(s)', fontsize=16)

plt.show();
#==============================================================================

