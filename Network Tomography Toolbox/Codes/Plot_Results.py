#=======================IMPORT THE NECESSARY LIBRARIES=========================
import numpy as np
import sys,getopt,os
from time import time
import matplotlib.pyplot as plt
import copy
from scipy.cluster.vq import whiten
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.cluster.vq import vq
from scipy import signal
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
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:D:A:F:R:L:M:B:R:G:J:K:U:Z:Y:O:o:f:V:p:j:b:")

file_name_ground_truth,file_name_spikes,ternary_mode,file_name_base_results,inference_method,sparsity_flag,beta,alpha0,no_itr_over_dataset,Var1_range,plot_flags,Var2_range,plot_vars,bin_size,no_processes,block_size,neuron_range,class_sample_freq,kernel_choice = parse_commands_ternary_algo(input_opts)

sparse_thr0 = 1.0                                    # The initial sparsity soft-threshold (not relevant in this version)
num_process = min(no_processes,multiprocessing.cpu_count())
block_size = min(block_size,T)

if len(neuron_range)>1:
    neuron_range = range(neuron_range[0],neuron_range[1])
#==============================================================================

#==================DO SANITY CHECK ON THE ENTERED PARAMETERS===================
if not len(neuron_range):
    print 'Sorry you should specify a list of neurons to plot over'
    sys.exit()

if not (len(Var1_range) or len(Var2_range)):
    print 'Sorry you should specify some variables to plot!'
    sys.exit()
else:
    if len(Var2_range) and (not len(Var1_range)):
        Var1_range = Var2_range
        Var2_range = [-1]
        plot_vars.append('aaa')
    
if not file_name_ground_truth:
    print 'Sorry you should specify a file that contains ground truth'
    sys.exit()
#==============================================================================

#================================INITIALIZATIONS===============================

#----------------------------Read and Sort Spikes------------------------------
if not file_name_spikes:
    file_name_spikes = '../Data/Spikes/Moritz_Spike_Times.txt'
    file_name_spikes = '/scratch/salavati/NeuralNetworkTomography/Network Tomography Toolbox/Data/Spikes/Moritz_Spike_Times.txt'
    #file_name_spikes = '../Data/Spikes/HC3_ec013_198_processed.txt'
    #file_name_spikes = '/scratch/salavati/NeuralNetworkTomography/Network Tomography Toolbox/Data/Spikes/HC3_ec013_198_processed.txt'
    
try:
    ll = file_name_spikes.split('/')
except:
    ll = file_name_spikes.split('/')
    
ll = ll[-1]
file_name_prefix = ll.split('.txt')
file_name_prefix = file_name_prefix[0]
#------------------------------------------------------------------------------




#------Calculate the Range to Assess the Effect of Recording Duration------
#Var1_range = np.array([325000,625000,925000,1225000])
plot_vars = ['T']

Var1_range = np.array(Var1_range)
Var2_range = np.array(Var2_range)
#--------------------------------------------------------------------------

#--------------------------Initialize Other Variables--------------------------
whiten_flag = 1                     # If 1, the algorithm whitens the inferred graph before calculating the results
zero_diagonals_flag = 1             # If 1, the diagonal elements (self feedback-loops) will be set to 0 before calculating belief qualities
adj_fact_exc = 0.75                 # This is the adjustment factor for clustering algorithms (between 0 and infinity).
adj_fact_inh = 0.5                  # This is the adjustment factor for clustering algorithms (between 0 and infinity).
parallel_flag = 1                   # If 1, the code reads the incoming connections to each neuron from a separate file
sparsity_flag = 5
get_from_hots = 1

if inference_method == 1:
    algorithm_name = 'Stochastic NeuInf'
elif inference_method == 4:        
    algorithm_name = 'Cross Correlogram'
elif inference_method == 8:
    algorithm_name = 'GLM'
elif inference_method == 7:
    algorithm_name = 'MSE'
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

#----------------------------Read The Actual Graph-----------------------------    
#file_name = '../Data/Graphs/Moritz_Actual_Connectivity.txt'
#file_name = '../Results/Inferred_Graphs/W_Pll_Moritz_I_7_S_5_T_75000_0.txt'
W = np.genfromtxt(file_name_ground_truth, dtype=None, delimiter='\t')
n,m = W.shape
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
if file_name_prefix == 'HC3':
    file_name_W_deg = '../../Code to Parse Other Datasets/HC-3/W_deg.txt'
    W_deg = np.genfromtxt(file_name_W_deg, dtype=None, delimiter='\t')
    eta = 0.5
    eta_max = 0.1
    
    neurons_type_sum = np.zeros([len(Var1_range),n])
    neurons_type_max = np.zeros([len(Var1_range),n])
    acc_neurons_type_sum = np.zeros([len(Var1_range),1])
    acc_neurons_type_max = np.zeros([len(Var1_range),1])
    false_neurons_type_sum = np.zeros([len(Var1_range),1])
    false_neurons_type_max = np.zeros([len(Var1_range),1])
    neurons_type_actual = np.zeros([n,1])
    for ik in range(0,n):
        if W_deg[ik,0] > W_deg[ik,1]:
            neurons_type_actual[ik,0] = 1
        elif W_deg[ik,0] < W_deg[ik,1]:
            neurons_type_actual[ik,0] = -1
else:
    mean_exc,mean_inh,mean_void,std_void_r,std_exc,std_inh,std_void,mean_void_r,Prec_exc,Prec_inh,Prec_void,Rec_exc,Rec_inh,Rec_void,std_Prec_exc,std_Prec_inh,std_Prec_void,std_Rec_exc,std_Rec_inh,std_Rec_void = initialize_plotting_variables(Var2_range,m)
    
    mean_exc = np.zeros([len(Var1_range),len(Var2_range)])
    std_exc = np.zeros([len(Var1_range),len(Var2_range)])
    
    mean_inh = np.zeros([len(Var1_range),len(Var2_range)])
    std_inh = np.zeros([len(Var1_range),len(Var2_range)])
    
    mean_void = np.zeros([len(Var1_range),len(Var2_range)])
    std_void = np.zeros([len(Var1_range),len(Var2_range)])
#------------------------------------------------------------------------------

#----------------------Read Belief Values From the File------------------------
itr_V1 = 0
for v1 in Var1_range:
    
    exec('%s = v1' %plot_vars[0])
    itr_V2 = 0
        
    for v2 in Var2_range:
            
        exec('%s = v2' %plot_vars[1])    
        
        W_inferred = np.zeros([n,m]) 
        for n_ind in neuron_range:
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~~~~~~~~~
            file_name_ending = 'I_' + str(inference_method) + '_S_' + str(sparsity_flag*sparse_thr0) + '_T_' + str(T)
            file_name_ending = file_name_ending + '_C_' + str(num_process) + '_B_' + str(block_size)
            file_name_ending = file_name_ending + '_K_' + kernel_choice + '_H_' + str(class_sample_freq)
            #file_name_ending = file_name_ending + '_F_' + str(class_sample_freq) 
        
            if bin_size:
                file_name_ending = file_name_ending + '_bS_' + str(bin_size)
            
            file_name_ending = file_name_ending + '_ii_' + str(no_itr_over_dataset)
            
            file_name =  file_name_base_results + "/Inferred_Graphs/W_Pll_%s_%s_%s.txt" %(file_name_prefix,file_name_ending,str(n_ind))

            file_name = file_name_base_results + "/Inferred_Graphs/W_Pll_%s_%s_%s.txt" %(file_name_prefix,file_name_ending,str(n_ind))
            
            if get_from_host:
                cmd = 'scp salavati@castor.epfl.ch:"~/NeuralNetworkTomography/Network\ Tomography\ Toolbox/Results/Inferred_Graphs/W_Pll_%s_%s_%s.txt" ../Results/Inferred_Graphs/' %(file_name_prefix,file_name_ending,str(ik))                
                os.system(cmd)
                #
            try:
                W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                #W_read = W_read/np.abs(W_read[:-1]).max()
                W_inferred[0:len(W_read),n_ind] = W_read
                print "Got the file!"
            except:
                found_file_tot = 0
                pdb.set_trace()
                print 'Sorry I can not find the corresponding inference file for the netwrok'
                
        
        
        for i in range (0,n):
            for j in range(0,m):
                if np.isnan(W_inferred[i,j]):
                    W_inferred[i,j] =0
                    
            #W_inferred = W_inferred + np.random.rand(n,m)/1000
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        

        
        #~~~~~~~~~~~~~~~~~~~~~~~~Claculate Belief Quality~~~~~~~~~~~~~~~~~~~~~~
        means_vector,std_vector = calculate_belief_quality(W_inferred,W,whiten_flag,zero_diagonals_flag)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        #~~~~~~~~~~~~~~~~~~~~~~~~Update the Variables~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_exc[itr_V1,itr_V2] = means_vector[0].mean()
        std_exc[itr_V1,itr_V2]  = std_vector[0].mean()
                                
        mean_inh[itr_V1,itr_V2] = means_vector[1].mean()
        std_inh[itr_V1,itr_V2] = std_vector[1].mean()
                                
        mean_void[itr_V1,itr_V2] = means_vector[2].mean()
        std_void[itr_V1,itr_V2] = std_vector[2].mean()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
        itr_V2 = itr_V2 + 1
            
    itr_V1 = itr_V1 + 1
#------------------------------------------------------------------------------

#--------------------------Read Precision and Recall---------------------------
if 0:
#?? fix this part: /W_Pll_%s_%s_%s.txt" %(file_name_prefix,file_name_ending,str(n_ind))
    file_name_ending2 = file_name_ending + "_%s" %str(adj_fact_exc)
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
        print 'Sorry I can not find the corresponding precision/recall file for the netwrok. Moving on...'
    
                        
    if found_file:
        #~~~~~~~~~~~Update Precision and Recall Variables~~~~~~~~~~~
        Prec_exc = (precision_tot[:,1]).T
        Prec_inh = (precision_tot[:,2]).T
        Prec_void = (precision_tot[:,3]).T
                        
        Rec_exc = (recall_tot[:,1]).T
        Rec_inh = (recall_tot[:,2]).T
        Rec_void = (recall_tot[:,3]).T
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
#------------------------------------------------------------------------------

#==============================================================================


#=================================PLOT THE RESULTS=============================

#----------------------------Plot the Blief Results----------------------------
if 'B' in plot_flags:
    
    if plot_vars[1] == 'p_miss':
        Var2_range = np.divide(1,Var2_range.astype(float))
    elif plot_vars[0] == 'p_miss':
        Var1_range = np.divide(1,Var1_range.astype(float))
            
    if (len(Var2_range) == 1) or (len(Var1_range) == 1):
        fig, axs = plt.subplots(nrows=1, ncols=1)
        ax = axs
        
        if (len(Var2_range) == 1):
            ax.errorbar(Var1_range,mean_exc[:,0],std_exc[:,0],color='r',label='Excitatory')        
            ax.errorbar(Var1_range,mean_inh[:,0],std_inh[:,0],color='b',label='Inhibitory');
            ax.errorbar(Var1_range,mean_void[:,0],std_void[:,0],color='g',label='Void');
            
            plt.xlabel(plot_vars[0], fontsize=16)
        else:
            ax.errorbar(Var2_range,mean_exc[0,:],std_exc[0,:],color='r',label='Excitatory')        
            ax.errorbar(Var2_range,mean_inh[0,:],std_inh[0,:],color='b',label='Inhibitory');
            ax.errorbar(Var2_range,mean_void[0,:],std_void[0,:],color='g',label='Void');
            
            plt.xlabel(plot_vars[1], fontsize=16)    
        
        
        plt.ylabel('Average of beliefs', fontsize=16)
        plt.legend(loc='lower left')
        ax.set_title('Average belief qualities for algorithm %s' %(algorithm_name))
        plt.show();
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        X, Y = np.meshgrid(Var2_range, Var1_range)
        
        surf = ax.plot_surface(X,Y, mean_exc,cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_xlabel(r'$\epsilon$', fontsize=16)
        ax.set_ylabel(r'$\sigma$', fontsize=16)    
        ax.set_zlabel('Mean value of excitatory links', fontsize=16)
    
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        surf = ax.plot_surface(X, Y, mean_void,cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_xlabel(r'$\epsilon$', fontsize=16)
        ax.set_ylabel(r'$\sigma$', fontsize=16)    
        ax.set_zlabel('Mean value of void links', fontsize=16)
        
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        surf = ax.plot_surface(X, Y, mean_inh,cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_xlabel(r'$\epsilon$', fontsize=16)
        ax.set_ylabel(r'$\sigma$', fontsize=16)    
        ax.set_zlabel('Mean value of inhibitory links', fontsize=16)
        
        #ax.set_title('Average belief qualities for algorithm %s' %(algorithm_name))
        
        plt.show()
    
    #plt.plot(mean_exc-mean_void);plt.show()
    if plotly_flag:
        plot_legends = ['Excitatory','Inhibitory','Void']
        plot_colors = ['#F62817','#1569C7','#4CC552']
        error_bar_colors = ['#F75D59','#368BC1','#54C571']
        
        x_array = np.vstack([Var2_range,Var2_range,Var2_range])
        y_array = np.vstack([mean_exc,mean_inh,mean_void])
        error_array = np.vstack([std_exc,std_inh,std_void])                
        no_plots = 3
                
        plot_title = 'Average belief qualities for algorithm %s' %(algorithm_name)
        plot_url = export_to_plotly(x_array,y_array,no_plots,plot_legends,'line',plot_colors,'t(s)','Average of beliefs',plot_title,error_array,error_bar_colors)
#------------------------------------------------------------------------------
            
#----------------------Plot the Precision/Recall Results=----------------------
if 'P' in plot_flags:
    bar_width = 0.15
    Var1_range = Var1_range/1000.0
    plt.bar(Var1_range,Prec_exc,bar_width,color='r',label='Exc. Precision');    
    plt.bar(Var1_range + bar_width,Prec_inh,bar_width,color='b',label='Inh. Precision');
    plt.bar(Var1_range + 2*bar_width,Prec_void,bar_width,color='g',label='Void Precision');
    plt.bar(Var1_range + 3*bar_width,Rec_exc,bar_width,color='red',edgecolor='black',hatch='//',label='Exc. Recall',);
    plt.bar(Var1_range + 4*bar_width,Rec_inh,bar_width,color='blue',edgecolor='black',hatch='//',label='Inh. Recall');
    plt.bar(Var1_range + 5*bar_width,Rec_void,bar_width,color='green',edgecolor='black',hatch='//',label='Void Recall');
            
    plt.title('Precision for algorithm %s' %(algorithm_name))
    plt.xlabel('t(s)', fontsize=16)
    plt.ylabel('Precision/Recall', fontsize=16)
    plt.legend(loc='upper left')
    plt.show();
        
    if plotly_flag:
        plot_legends = ['Exc. Precision','Inh. Precision','Void Precision']
        plot_colors = ['#F62817','#1569C7','#4CC552']
        #error_bar_colors = ['#F75D59','#368BC1','#54C571']
            
        x_array = np.vstack([Var1_range,Var1_range,Var1_range])
        y_array = np.vstack([Prec_exc,Prec_inh,Prec_void])
        #error_array = np.vstack([std_exc,std_inh,std_void])
        no_plots = 3
                
        plot_title = 'Precision for algorithm %s' %(algorithm_name)
        plot_url = export_to_plotly(x_array,y_array,no_plots,plot_legends,'bar',plot_colors,'t(s)','Precision',plot_title)
#------------------------------------------------------------------------------
            
#-----------------------Plot the Scatter of Beliefs----------------------------
if found_file_tot:    
    if whiten_flag:
        n,m = W_inferred.shape
        W_inferred = W_inferred + np.random.rand(n,m)/100000
        W_inferred  = whiten(W_inferred)
         
    if 'S' in plot_flags:
        plt.title('Scatter plot of belief')
        plt.scatter(np.sign(W.ravel()),W_inferred.ravel())
        plt.xlabel('G (actual)', fontsize=16)
        plt.ylabel('W (inferred)', fontsize=16)
        plt.show()
    else:
        W_inferred = np.array([0])
#------------------------------------------------------------------------------

#-------------------------------Save the Results-------------------------------
if 'B' in plot_flags:
    save_plot_results(Var1_range,mean_exc,std_exc,mean_inh,std_inh,mean_void,
                          std_void,mean_void_r,std_void_r,file_name_base_results,
                          file_name_ending,0,W_inferred,W)
        
if 'P' in plot_flags:
    file_name_ending = file_name_ending.replace('_T_%s'%str(T),'')        
    save_precision_recall_results(Var1_range,file_name_base_results,file_name_ending,adj_fact_exc,adj_fact_inh,ternary_mode,
                                      Prec_exc,std_Prec_exc,Prec_inh,std_Prec_inh,Prec_void,
                                      std_Prec_void,Rec_exc,std_Rec_exc,Rec_inh,std_Rec_inh,Rec_void,std_Rec_void)

#------------------------------------------------------------------------------

#==============================================================================


#=================SAVE RESULTS FOR WEB DEMO IF NECESSARY=======================
pdb.set_trace()
if 0:
    save_web_demo(W,W_inferred,file_name_base_results,file_name_ending)
#==============================================================================


file_name_ending =  file_name_prefix + '_I_' + str(inference_method) + '_S_' + str(sparsity_flag) + '_T_' + str(T)
if p_miss:
    file_name_ending = file_name_ending + '_pM_' + str(p_miss)
if jitt:
    file_name_ending = file_name_ending + '_jt_' + str(jitt)
if bin_size:
    file_name_ending = file_name_ending + '_bS_' + str(bin_size)
            
file_name =  file_name_base_results + "/Plot_Results/Neuron_Type_%s.txt" %(file_name_ending)
temp = np.hstack([acc_neurons_type_sum,false_neurons_type_sum,acc_neurons_type_max,false_neurons_type_max])
T_range = np.reshape(T_range,[len(T_range),1])
T_range = np.divide(T_range,1000).astype(int)
temp = np.hstack([T_range,temp])
np.savetxt(file_name,temp,'%2.5f',delimiter='\t')


# np.savetxt('./Y_predict_t_inds.txt',Y_predict,'%2.5f',delimiter='\t')