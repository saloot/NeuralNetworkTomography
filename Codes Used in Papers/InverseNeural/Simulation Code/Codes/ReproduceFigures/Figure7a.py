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
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:J:K:U:Z:Y:O:f:p:q:c:l:")
frac_stimulated_neurons,T_max,ensemble_size,file_name_base_data,ensemble_count_init,generate_data_mode,ternary_mode,file_name_base_results,inference_method,sparsity_flag,we_know_topology,beta,alpha0,infer_itr_max,T_range,plot_flags = parse_commands_ternary_algo(input_opts)

p_list = []                                     # The list of connection probabilities to plot
q_list = []                                     # The list of stimulation probabilities to plot
colors_list = []                                # List of the colors for each bar chart
plot_list = []                                  # The number of bar charts in each plot

if (input_opts):
    for opt, arg in input_opts:
        if opt == '-p':
            temp = (arg).split(',')
            for i in temp:                        
                p_list.append(float(i))                
        
        if opt == '-q':
            temp = (arg).split(',')
            for i in temp:                        
                q_list.append(float(i))
        
        if opt == '-c':
            temp = (arg).split(',')
            for i in temp:                        
                colors_list.append(i)
        
        if opt == '-l':
            temp = (arg).split(',')
            for i in temp:                        
                plot_list.append(int(i))
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
gap_exc_void = {}
gap_void_inh = {}
T_range_list = {}

for i in range(0,len(p_list)):
    p = p_list[i]
    
    frac_stimulated_neurons = q_list[i]
    Network.connection_prob_matrix[0,1] = p
    Network.read_weights(0,file_name_base_data)
    T = T_range[i]

    file_name_ending = generate_file_name(Network.file_name_ending,3,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')        
    file_name_ending = file_name_ending + '_1' 
    file_name = file_name_base_results + "/Plot_Results/Gap_mean_exc_void_%s.txt" %file_name_ending

    temp = np.genfromtxt(file_name, dtype='float', delimiter='\t')
    temp = temp[:,1]
    gap_exc_void[str(i)] = temp[-1:]

    file_name = file_name_base_results + "/Plot_Results/Gap_mean_void_inh_%s.txt" %file_name_ending
    temp = np.genfromtxt(file_name, dtype='float', delimiter='\t')

    T_range_list[str(i)] = temp[:,0]
    temp = temp[:,1]
    gap_void_inh[str(i)] = temp[-1:]
#------------------------------------------------------------------------------


#==============================================================================



#=================================PLOT THE RESULTS=============================
bar_width = 0.1
fig, axs = plt.subplots(nrows=len(plot_list), ncols=1)

T_range_temp = np.array([0,1])

ind1 = 0
i_so_far = 0 
for j in range(0,len(plot_list)):
    
    ax = axs[j]
    
    ind2 = ind1 + plot_list[j]
    pp_list = p_list[ind1:ind2]
    
    for i in range(0,len(pp_list)):
        if j == 0:
            leg = 'p = ' + str(p_list[i_so_far])
        else:
            leg = 'q = ' + str(q_list[i_so_far])
            
        gap_temp = [gap_void_inh[str(i_so_far)],gap_exc_void[str(i_so_far)]]
        
        ax.bar(T_range_temp + int(i)*bar_width,gap_temp,bar_width,color=colors_list[i_so_far],label=leg);
        i_so_far = i_so_far + 1
        
    if (j == 0):
        ax.set_xticks(T_range_temp+0.15)
        ax.set_xticklabels( ('void-inh','exc-void') )
        ax.set_title('q = 0.3, T = 5.5s')
        ax.grid()
        ax.legend(loc='upper right',prop={'size':12})
    elif (j == 1):
        ax.set_xticks(T_range_temp+0.1)
        ax.set_xticklabels( ('void-inh','exc-void') )
        ax.set_title('p = 0.3, T = 5.5s')
        ax.grid()
        ax.legend(loc='upper right',prop={'size':12})


plt.show();
#==============================================================================

