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

from CommonFunctions.Neurons_and_Networks import *
from CommonFunctions.default_values import *
from CommonFunctions.auxiliary_functions_digitize import caculate_accuracy,parse_commands_ternary_algo
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

#--------------------------Initialize Other Variables--------------------------
whiten_flag = 1                     # If 1, the algorithm whitens the inferred graph before calculating the results
zero_diagonals_flag = 1             # If 1, the diagonal elements (self feedback-loops) will be set to 0 before calculating belief qualities
adj_fact_exc = 0.75                 # This is the adjustment factor for clustering algorithms (between 0 and infinity).
adj_fact_inh = 0.5                  # This is the adjustment factor for clustering algorithms (between 0 and infinity).

if inference_method == 3:
    algorithm_name = 'Stochastic NeuInf'
elif inference_method == 4:        
    algorithm_name = 'Cross Correlogram'
elif inference_method == 8:
    algorithm_name = 'GLM'
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
mean_exc,mean_inh,mean_void,std_void_r,std_exc,std_inh,std_void,mean_void_r,Prec_exc,Prec_inh,Prec_void,Rec_exc,Rec_inh,Rec_void,std_Prec_exc,std_Prec_inh,std_Prec_void,std_Rec_exc,std_Rec_inh,std_Rec_void = initialize_plotting_variables(Network,we_know_topology,T_range,ensemble_size,ensemble_count_init)
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
            try:
                W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                found_file_tot = 1
            except:
                found_file_tot = 0
                print 'Sorry I can not find the corresponding inference file for the netwrok'
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
                    try:
                        W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                        found_file = 1
                    except:
                        found_file = 0
                        print 'Sorry I can not find the corresponding inference file for the sub-netwrok from layer %s to %s. Moving on...' %(str(l_in),str(l_out))
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    if found_file:
                        #~~~~~~~~~~~~~~~~~Claculate Belief Quality~~~~~~~~~~~~~~~~~~
                        means_vector,std_vector = calculate_belief_quality(Network,W_inferred_our,W,l_out,whiten_flag,zero_diagonals_flag,we_know_topology)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                        #~~~~~~~~~~~~Update the Belief Quality Variables~~~~~~~~~~~~
                        mean_exc[ind][itr_T,:] = mean_exc[ind][itr_T,:] + means_vector[0]
                        std_exc[ind][itr_T,:] = std_exc[ind][itr_T,:] + std_vector[0]
                        
                        mean_inh[ind][itr_T,:] = mean_inh[ind][itr_T,:] + means_vector[1]
                        std_inh[ind][itr_T,:] = std_inh[ind][itr_T,:] + std_vector[1]
                    
                        mean_void[ind][itr_T,:] = mean_void[ind][itr_T,:] + means_vector[2]
                        std_void[ind][itr_T,:] = std_void[ind][itr_T,:] + std_vector[2]
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~Read Precision and Recall~~~~~~~~~~~~~~~~~~
                    file_name_ending2 = file_name_ending.replace('_T_%s'%str(T),'')
                    file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
                    file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                    file_name_ending2 = file_name_ending2 + "_B_%s" %str(ternary_mode)
                    
                    try: 
                        file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %file_name_ending2
                        precision_tot = np.genfromtxt(file_name, dtype='float', delimiter='\t')
                
                        file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %file_name_ending2
                        recall_tot = np.genfromtxt(file_name, dtype='float', delimiter='\t')
                    
                        found_file = 1
                    except:
                        found_file = 0
                        print 'Sorry I can not find the corresponding precision/recall file for the sub-netwrok from layer %s to %s. Moving on...' %(str(l_in),str(l_out))
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    if found_file:
                        #~~~~~~~~~~~Update Precision and Recall Variables~~~~~~~~~~~                    
                        Prec_exc[ind][:,ensemble_count-ensemble_count_init] = (precision_tot[:,1]).T
                        Prec_inh[ind][:,ensemble_count-ensemble_count_init] = (precision_tot[:,2]).T
                        Prec_void[ind][:,ensemble_count-ensemble_count_init] = (precision_tot[:,3]).T
                    
                        Rec_exc[ind][:,ensemble_count-ensemble_count_init] = (recall_tot[:,1]).T
                        Rec_inh[ind][:,ensemble_count-ensemble_count_init] = (recall_tot[:,2]).T
                        Rec_void[ind][:,ensemble_count-ensemble_count_init] = (recall_tot[:,3]).T
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
            else:                    
                
                ind = str(l_out)
                
                if found_file_tot:
                    
                    #~~~~~~~~~~~~~~~~~Claculate Belief Quality~~~~~~~~~~~~~~~~~~
                    means_vector,std_vector = calculate_belief_quality(Network,W_inferred_our,W,l_out,whiten_flag,zero_diagonals_flag,we_know_topology)
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                    #~~~~~~~~~~~~~~~~~~~~Update the Variables~~~~~~~~~~~~~~~~~~~~~~~                    
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
        
                #~~~~~~~~~~~~~~~~Read Precision and Recall~~~~~~~~~~~~~~~~~~
                file_name_ending2 = file_name_ending + '_' + str(l_out)                
                file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
                file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
                file_name_ending2 = file_name_ending2 + "_B_%s" %str(ternary_mode)
                file_name_ending2 = file_name_ending2.replace('_T_%s'%str(T),'')
                
                try:
                    file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %file_name_ending2            
                    precision_tot = np.genfromtxt(file_name, dtype='float', delimiter='\t')
                
                    file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %file_name_ending2
                    recall_tot = np.genfromtxt(file_name, dtype='float', delimiter='\t')
                    found_file = 1
                except:
                    found_file = 0
                    print 'Sorry I can not find the corresponding precision/recall file for the sub-netwrok coming to layer %s. Moving on...' %(str(l_out))
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                if found_file:
                    #~~~~~~~~~~~Update Precision and Recall Variables~~~~~~~~~~~
                    Prec_exc[ind][:,ensemble_count-ensemble_count_init] = (precision_tot[:,1]).T
                    Prec_inh[ind][:,ensemble_count-ensemble_count_init] = (precision_tot[:,2]).T
                    Prec_void[ind][:,ensemble_count-ensemble_count_init] = (precision_tot[:,3]).T
                    
                    Rec_exc[ind][:,ensemble_count-ensemble_count_init] = (recall_tot[:,1]).T
                    Rec_inh[ind][:,ensemble_count-ensemble_count_init] = (recall_tot[:,2]).T
                    Rec_void[ind][:,ensemble_count-ensemble_count_init] = (recall_tot[:,3]).T
                    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        itr_T = itr_T + 1
    #--------------------------------------------------------------------------    

#==============================================================================


#=======================POST-PROCESS AND STORE THE RESULTS=====================
T_range = np.divide(T_range,1000.0)                 # Display the results in seconds (rather than mili seconds)
for l_out in range(0,Network.no_layers):
    if we_know_topology.lower() == 'y':
        for l_in in range(0,l_out+1):
            #~~~~~~~~~~~~Update the Belief Quality Variables~~~~~~~~~~~~
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
            
            #~~~~~~~~~~~~~~~Update Precision and Recall~~~~~~~~~~~~~~~~~
            std_Prec_exc[ind] = Prec_exc[ind].std(axis=1)
            std_Prec_inh[ind] = Prec_inh[ind].std(axis=1)
            std_Prec_void[ind] = Prec_void[ind].std(axis=1)
            std_Rec_exc[ind] = Rec_exc[ind].std(axis=1)
            std_Rec_inh[ind] = Rec_inh[ind].std(axis=1)
            std_Rec_void[ind] = Rec_void[ind].std(axis=1)
            
            Prec_exc[ind] = Prec_exc[ind].mean(axis=1)
            Prec_inh[ind] = Prec_inh[ind].mean(axis=1)
            Prec_void[ind] = Prec_void[ind].mean(axis=1)
            Rec_exc[ind] = Rec_exc[ind].mean(axis=1)
            Rec_inh[ind] = Rec_inh[ind].mean(axis=1)
            Rec_void[ind] = Rec_void[ind].mean(axis=1)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
            #~~~~~~~~~~~~~~~~~~Plot the Blief Results~~~~~~~~~~~~~~~~~~~
            if 'B' in plot_flags:
                plt.errorbar(T_range,mean_exc[ind],std_exc[ind],color='r',label='Excitatory');
                plt.errorbar(T_range,mean_inh[ind],std_inh[ind],color='b',label='Inhibitory');
                plt.errorbar(T_range,mean_void[ind],std_void[ind],color='g',label='Void');
            
                plt.title('Average belief qualities from layer %s to layer %s, for %s algorithm' %(str(l_in),str(l_out),algorithm_name))
                plt.xlabel('t(s)', fontsize=16)
                plt.ylabel('Average of beliefs', fontsize=16)
                plt.legend(loc='lower left')
                plt.show();            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #~~~~~~~~~~~~~Plot the Precision/Recall Results~~~~~~~~~~~~~
            if 'P' in plot_flags:
                bar_width = 0.35
                plt.bar(T_range,Prec_exc[ind],bar_width,color='r',label='Exc. Precision');
                plt.bar(T_range + bar_width,Prec_inh[ind],bar_width,color='b',label='Inh. Precision');
                plt.bar(T_range + 2*bar_width,Prec_void[ind],bar_width,color='g',label='Void Precision');
                plt.bar(T_range + 3*bar_width,Rec_exc[ind],bar_width,color='red',edgecolor='black',hatch='//',label='Exc. Recall',);
                plt.bar(T_range + 4*bar_width,Rec_inh[ind],bar_width,color='blue',edgecolor='black',hatch='//',label='Inh. Recall');
                plt.bar(T_range + 5*bar_width,Rec_void[ind],bar_width,color='green',edgecolor='black',hatch='//',label='Void Recall');
            
                plt.title('Precision and recall from layer %s to layer %s for %s algorithm' %(str(l_in),str(l_out),algorithm_name))
                plt.xlabel('t(s)', fontsize=16)
                plt.ylabel('Precision/Recall', fontsize=16)
                plt.legend(loc='lower left')
                plt.show();
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #~~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~~
            file_name_ending_base = Network.file_name_ending + '_l_' + str(l_in) + '_to_' + str(l_out)                
            file_name_ending = generate_file_name(file_name_ending_base,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')                    
            file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
            try:
                W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                found_file = 1                
            except:
                found_file = 0
                print 'Sorry I can not find the corresponding inference file for the sub-netwrok from layer %s to %s. Moving on...' %(str(l_in),str(l_out))
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            #~~~~~~~~~~~~~~~Plot the Scatter of Beliefs~~~~~~~~~~~~~~~~~
            if found_file:
                #~~~~~~~~~~~~~~~Extract the Weights for Each Layer~~~~~~~~~~~~~~~~
                ind = str(l_in) + str(l_out)
                temp_list = Network.Neural_Connections[ind]
                W = temp_list[0]
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
                if whiten_flag:
                    n,m = W_inferred_our.shape
                    W_inferred_our = W_inferred_our + np.random.rand(n,m)/100000
                    W_inferred_our  = whiten(W_inferred_our)
            
                if 'S' in plot_flags:
                    plt.title('Scatter plot of belief from layer %s to layer %s for %s algorithm' %(str(l_in),str(l_out),algorithm_name))
                    plt.scatter(np.sign(W.ravel()),W_inferred_our.ravel())
                    plt.xlabel('G (actual)', fontsize=16)
                    plt.ylabel('W (inferred)', fontsize=16)
                    plt.show()
            else:
                W_inferred_our = np.array([0])
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    
            #~~~~~~~~~~~~~~~~~~~~Save the Results~~~~~~~~~~~~~~~~~~~~~~~            
            save_plot_results(T_range,mean_exc[ind],std_exc[ind],mean_inh[ind],std_inh[ind],mean_void[ind],
                              std_void[ind],mean_void_r[ind],std_void_r[ind],
                              file_name_base_results,file_name_ending,0,W_inferred_our,W)
            
            file_name_ending_temp = file_name_ending.replace('_T_%s'%str(T),'')        
            save_precision_recall_results(T_range,file_name_base_results,file_name_ending_temp,adj_fact_exc,adj_fact_inh,ternary_mode,
                                      Prec_exc[ind],std_Prec_exc[ind],Prec_inh[ind],std_Prec_inh[ind],Prec_void[ind],
                                      std_Prec_void[ind],Rec_exc[ind],std_Rec_exc[ind],Rec_inh[ind],std_Rec_inh[ind],Rec_void[ind],std_Rec_void[ind])            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    else:                    
            
        #~~~~~~~~~~~Update the Belief Quality Variables~~~~~~~~~~~
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
            
            #plt.plot(mean_exc[ind]-mean_void[ind],'r');plt.plot(mean_inh[ind]-mean_void[ind],'b');plt.show()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~Update Precision and Recall~~~~~~~~~~~~~~~~~
        std_Prec_exc[ind] = Prec_exc[ind].std(axis=1)
        std_Prec_inh[ind] = Prec_inh[ind].std(axis=1)
        std_Prec_void[ind] = Prec_void[ind].std(axis=1)
        std_Rec_exc[ind] = Rec_exc[ind].std(axis=1)
        std_Rec_inh[ind] = Rec_inh[ind].std(axis=1)
        std_Rec_void[ind] = Rec_void[ind].std(axis=1)
        
        Prec_exc[ind] = Prec_exc[ind].mean(axis=1)
        Prec_inh[ind] = Prec_inh[ind].mean(axis=1)
        Prec_void[ind] = Prec_void[ind].mean(axis=1)
        Rec_exc[ind] = Rec_exc[ind].mean(axis=1)
        Rec_inh[ind] = Rec_inh[ind].mean(axis=1)
        Rec_void[ind] = Rec_void[ind].mean(axis=1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        #~~~~~~~~~~~~~~~~~~~Plot the Results~~~~~~~~~~~~~~~~~~~~~~~~
        if 'B' in plot_flags:
            fig, axs = plt.subplots(nrows=1, ncols=1)
            ax = axs
            ax.errorbar(T_range,mean_exc[ind],std_exc[ind],color='r',label='Excitatory')        
        
            ax.errorbar(T_range,mean_inh[ind],std_inh[ind],color='b',label='Inhibitory');
            ax.errorbar(T_range,mean_void[ind],std_void[ind],color='g',label='Void');
            if Network.no_layers > 1:
                ax.errorbar(T_range,mean_void_r[ind],std_void_r[ind],color='k',label='Void, Recurrent');
            
            ax.set_title('Average belief qualities for layer %s using %s algorithm' %(str(l_out),algorithm_name))
            plt.xlabel('t(s)', fontsize=16)
            plt.ylabel('Average of beliefs', fontsize=16)
            plt.legend(loc='lower left')
            plt.show();
            #pdb.set_trace()
        
            if plotly_flag:
                plot_legends = ['Excitatory','Inhibitory','Void']
                plot_colors = ['#F62817','#1569C7','#4CC552']
                error_bar_colors = ['#F75D59','#368BC1','#54C571']
                if Network.no_layers > 1:
                    x_array = np.vstack([T_range,T_range,T_range,T_range])
                    y_array = np.vstack([mean_exc[ind],mean_inh[ind],mean_void[ind],mean_void_r[ind]])
                    error_array = np.vstack([std_exc[ind],std_inh[ind],std_void[ind],std_void_r[ind]])
                    plot_legends.append('Void, Recurrent')
                    plot_colors.append('#B6B6B4')
                    error_bar_colors.append('#D1D0CE')
                    no_plots = 4
                
                else:
                    x_array = np.vstack([T_range,T_range,T_range])
                    y_array = np.vstack([mean_exc[ind],mean_inh[ind],mean_void[ind]])
                    error_array = np.vstack([std_exc[ind],std_inh[ind],std_void[ind]])                
                    no_plots = 3
                
                plot_title = 'Average belief qualities for layer %s using %s algorithm' %(str(l_out),algorithm_name)
                plot_url = export_to_plotly(x_array,y_array,no_plots,plot_legends,'line',plot_colors,'t(s)','Average of beliefs',plot_title,error_array,error_bar_colors)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
        #~~~~~~~~~~~~~Plot the Precision/Recall Results~~~~~~~~~~~~~
        if 'P' in plot_flags:
            bar_width = 0.35
            plt.bar(T_range,Prec_exc[ind],bar_width,color='r',label='Exc. Precision');
            plt.bar(T_range + bar_width,Prec_inh[ind],bar_width,color='b',label='Inh. Precision');
            plt.bar(T_range + 2*bar_width,Prec_void[ind],bar_width,color='g',label='Void Precision');
            plt.bar(T_range + 3*bar_width,Rec_exc[ind],bar_width,color='red',edgecolor='black',hatch='//',label='Exc. Recall',);
            plt.bar(T_range + 4*bar_width,Rec_inh[ind],bar_width,color='blue',edgecolor='black',hatch='//',label='Inh. Recall');
            plt.bar(T_range + 5*bar_width,Rec_void[ind],bar_width,color='green',edgecolor='black',hatch='//',label='Void Recall');
            
            plt.title('Precision for layer %s using %s algorithm ' %(str(l_out),algorithm_name))
            plt.xlabel('t(s)', fontsize=16)
            plt.ylabel('Precision/Recall', fontsize=16)
            plt.legend(loc='upper left')
            plt.show();
        
            if plotly_flag:
                plot_legends = ['Exc. Precision','Inh. Precision','Void Precision']
                plot_colors = ['#F62817','#1569C7','#4CC552']
                #error_bar_colors = ['#F75D59','#368BC1','#54C571']
            
                x_array = np.vstack([T_range,T_range,T_range])
                y_array = np.vstack([Prec_exc[ind],Prec_inh[ind],Prec_void[ind]])
                #error_array = np.vstack([std_exc[ind],std_inh[ind],std_void[ind]])                
                no_plots = 3
                
                plot_title = 'Precision for layer %s using %s algorithm ' %(str(l_out),algorithm_name)
                plot_url = export_to_plotly(x_array,y_array,no_plots,plot_legends,'bar',plot_colors,'t(s)','Precision',plot_title)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~
        file_name_ending = generate_file_name(Network.file_name_ending,inference_method,we_know_topology,'A',generate_data_mode,infer_itr_max,frac_stimulated_neurons,sparsity_flag,T,'N')        
        file_name = file_name_base_results + "/Inferred_Graphs/W_%s.txt" %file_name_ending
        try:
            W_inferred_our = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            found_file = 1
        except:
            found_file = 0
            print 'Sorry I can not find the corresponding inference file for the sub-netwrok coming to layer %s. Moving on...' %(str(l_out))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if found_file:
        #~~~~~~~~~~~~~~Plot the Scatter of Beliefs~~~~~~~~~~~~~~~~
            if whiten_flag:
                n,m = W_inferred_our.shape
                W_inferred_our = W_inferred_our + np.random.rand(n,m)/100000
                W_inferred_our  = whiten(W_inferred_our)
        
            if 'S' in plot_flags:
                plt.title('Scatter plot of beliefs %s for %s algorithm' %(str(l_out),algorithm_name))
                plt.scatter(np.sign(W.ravel()),W_inferred_our.ravel())
                plt.xlabel('G (actual)', fontsize=16)
                plt.ylabel('W (inferred)', fontsize=16)
                plt.show()
        else:
            W_inferred_our = np.array([0])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        #~~~~~~~~~~~~~~~~~~~~Save the Results~~~~~~~~~~~~~~~~~~~~~~~
        file_name_ending_temp = file_name_ending + '_' + str(l_out)
        save_plot_results(T_range,mean_exc[ind],std_exc[ind],mean_inh[ind],std_inh[ind],mean_void[ind],
                          std_void[ind],mean_void_r[ind],std_void_r[ind],file_name_base_results,
                          file_name_ending_temp,in_recurrent_flag,W_inferred_our,W)
        
        file_name_ending_temp = file_name_ending_temp.replace('_T_%s'%str(T),'')        
        save_precision_recall_results(T_range,file_name_base_results,file_name_ending_temp,adj_fact_exc,adj_fact_inh,ternary_mode,
                                      Prec_exc[ind],std_Prec_exc[ind],Prec_inh[ind],std_Prec_inh[ind],Prec_void[ind],
                                      std_Prec_void[ind],Rec_exc[ind],std_Rec_exc[ind],Rec_inh[ind],std_Rec_inh[ind],Rec_void[ind],std_Rec_void[ind])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
#==============================================================================


#=================SAVE RESULTS FOR WEB DEMO IF NECESSARY=======================
if 0:
    save_web_demo(W,W_inferred_our,file_name_base_results,file_name_ending)
#==============================================================================
