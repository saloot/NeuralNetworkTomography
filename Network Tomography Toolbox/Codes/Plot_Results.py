#=======================IMPORT THE NECESSARY LIBRARIES=========================
import numpy as np
import sys,getopt,os
from time import time
import matplotlib.pyplot as plt
import copy
from scipy.cluster.vq import whiten
#import matplotlib.mlab as mlab
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from scipy.cluster.vq import vq
#from scipy import signal
try: 
    import plotly.plotly as pltly
    from plotly.graph_objs import *
    plotly_import = 1
except:
    print 'Plotly was not found. No problem though, life goes on ;)'
    plotly_import = 0

from CommonFunctions.auxiliary_functions_plot import save_plot_results,calculate_belief_quality,save_web_demo,initialize_plotting_variables,save_precision_recall_results,export_to_plotly,parse_commands_plots

os.system('clear')                                              # Clear the commandline window
#==============================================================================

#==========================PARSE COMMAND LINE ARGUMENTS========================
input_opts, args = getopt.getopt(sys.argv[1:],"hE:I:P:Q:T:S:A:F:R:L:M:B:R:G:J:N:U:Z:Y:O:o:f:V:p:x:y:n:")

file_name_ending_list,file_name_base_results,file_name_ground_truth,x_label,y_label,plot_type,plot_var,x_axis_values,network_size,n_ind,no_hidden_neurons,no_structural_connections = parse_commands_plots(input_opts)
#==============================================================================

#==================DO SANITY CHECK ON THE ENTERED PARAMETERS===================
if len(x_axis_values):
    x_axis_values = np.array(x_axis_values)

if plot_type == 'W' and not file_name_ground_truth:
    print 'Sorry! To plot the quality of beliefs, you must specify the file that contains the ground truth'
    sys.exit()
elif file_name_ground_truth:
    W = np.genfromtxt(file_name_ground_truth, dtype=None, delimiter='\t')
    W = W.T
    n,m = W.shape
    W_ss = W[:,n_ind]
    W_s = np.zeros([n-no_hidden_neurons-no_structural_connections,1])
#==============================================================================

#================================INITIALIZATIONS===============================

#--------------------------Initialize Other Variables--------------------------
whiten_flag = 1                     # If 1, the algorithm whitens the inferred graph before calculating the results
zero_diagonals_flag = 1             # If 1, the diagonal elements (self feedback-loops) will be set to 0 before calculating belief qualities
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

#---------------------Initialize Simulation Variables--------------------------
itr = 0

vals_exc = np.zeros([len(x_axis_values)])
std_exc = np.zeros([len(x_axis_values)])

vals_inh = np.zeros([len(x_axis_values)])
std_inh = np.zeros([len(x_axis_values)])

vals_void = np.zeros([len(x_axis_values)])
std_void = np.zeros([len(x_axis_values)])


#================Plot the Precision/Recall Results================
if plot_type in ['P','R']:

    #------------------------Read the Files-----------------------
    itr = 0
    for file_name_ending in file_name_ending_list:
        if plot_type == 'P':
            file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %file_name_ending
        elif plot_type == 'R':
            file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %file_name_ending
            

        #~~~~~~~~~Update Precision and Recall Variables~~~~~~~~~~~
        temp_str = plot_var + '_' + str(x_axis_values[itr])
        if temp_str not in file_name:
            print('Something seems to be wrong!')
        else:
            vals = np.genfromtxt(file_name, dtype='float', delimiter='\t')
            vals_exc[itr] = vals[0]
            vals_inh[itr] = vals[1]
            vals_void[itr] = vals[2]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            itr += 1
            if itr >= len(x_axis_values):
                break
    #-------------------------------------------------------------

    #-----------------------Plot the Results-----------------------
    bar_width = 0.15
    #x_axis_values = x_axis_values/1000.0
    plt.bar(x_axis_values,vals_exc,bar_width,color='r',label='Excitatory');    
    plt.bar(x_axis_values + bar_width,vals_inh,bar_width,color='b',label='Inhibitory');
    plt.bar(x_axis_values + 2*bar_width,vals_void,bar_width,color='g',label='Void');
            
    if plot_type == 'P':
        plt.title('Precision')
    elif plot_type == 'R':
        plt.title('Recall')

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc='upper left')
    plt.show();
    #-------------------------------------------------------------

    #----------------------Save the Plots-------------------------

    #-------------------------------------------------------------

    #------------------Export Plots to Plotly---------------------
    if plotly_flag:
        plot_legends = ['Exc. Precision','Inh. Precision','Void Precision']
        plot_colors = ['#F62817','#1569C7','#4CC552']
        #error_bar_colors = ['#F75D59','#368BC1','#54C571']
            
        x_array = np.vstack([x_axis_values,x_axis_values,x_axis_values])
        y_array = np.vstack([vals_exc,vals_inh,vals_void])
        #error_array = np.vstack([std_exc,std_inh,std_void])
        no_plots = 3
                
        plot_title = 'Precision'
        plot_url = export_to_plotly(x_array,y_array,no_plots,plot_legends,'bar',plot_colors,'t(s)','Precision',plot_title)
    #-------------------------------------------------------------
#=================================================================


#================Plot Results on Spend Resources==================
if plot_type == 'C':
    spent_cpu = np.zeros([len(x_axis_values)])
    spent_ram = np.zeros([len(x_axis_values)])

    #------------------------Read the Files-----------------------
    itr = 0
    for file_name_ending in file_name_ending_list:
    
        file_name = file_name_base_results + '/Spent_Resources/CPU_RAM_' + file_name_ending
        file_name = file_name.replace('W_Pll_','')
        temp = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        temp_str = plot_var + '_' + str(x_axis_values[itr])
        if temp_str not in file_name:
            print('Something seems to be wrong!')
        spent_cpu[itr] = temp[1]
        spent_ram[itr] = temp[2]

        itr += 1
        if itr >= len(x_axis_values):
            break
    #-------------------------------------------------------------

    #-----------------------Plot the Results-----------------------
    plt.plot(x_axis_values,spent_cpu,color='r')
    plt.title('CPU Time')
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('T(s)', fontsize=16)
    plt.show()

    plt.plot(x_axis_values,spent_ram,color='r')
    plt.title('RAM Usage')
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('MB', fontsize=16)
    plt.show()
    #-------------------------------------------------------------

#=================================================================


#========Plot the Quality of Beliefs (Average of Weights)=========
if plot_type == 'W':
    means_vector = np.zeros([3,len(x_axis_values)])
    std_vector = np.zeros([3,len(x_axis_values)])

    itr_x = 0
    for x in x_axis_values:
    #------------------Read the Filesin the Ensemble---------------
        itr_i = 0

        file_name_temp = plot_var + '_' + str(x)
        for file_name_ending in file_name_ending_list:
            if file_name_temp in file_name_ending:    
                file_name = file_name_base_results + '/Inferred_Graphs/' + file_name_ending
                W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            else:
                continue
    #-------------------------------------------------------------

    #--------Reconstruct the Ground Truth in This Ensemble--------
            W_r = np.reshape(W[:,n_ind],[len(W[:,n_ind]),1])
            W_s = W_read[0:min(network_size,len(W_read))]

            if no_hidden_neurons or no_structural_connections:
                file_name_ending_mod = file_name_ending.replace('W_Pll_','')
                
                file_name_hidden = "Inferred_Graphs/Hidden_or_Structured_Neurons_" + file_name_ending_mod
                file_name = file_name_base_results + '/' + file_name_hidden
                hidden_neurons = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                W_r = np.delete(W_r,hidden_neurons,0)
                        
            W_s = np.reshape(W_s,[len(W_s),1])
            W_r = np.reshape(W_r,[len(W_r),1])
    #-------------------------------------------------------------

    #------------Calculate Mean and Variance of Beliefs-----------
            W_e = np.ma.masked_array(W_s,mask= (W_r<=0).astype(int))
            means_vector[0,itr_x] += W_e.mean()#.data
            std_vector[0,itr_x] += W_e.std()#.data
            
            W_i = np.ma.masked_array(W_s,mask= (W_r>=0).astype(int))
            means_vector[1,itr_x] += W_i.mean()#.data
            std_vector[1,itr_x] += W_i.std()#.data
                            
            W_v = np.ma.masked_array(W_s,mask= (W_r!=0).astype(int))
            means_vector[2,itr_x] += W_v.mean()#.data
            std_vector[2,itr_x] += W_v.std()#.data

            itr_i += 1

        means_vector[:,itr_x] /= float(itr_i)            # Normalize w.r.t. ensemble size
        std_vector[:,itr_x] /= float(itr_i)              # Normalize w.r.t. ensemble size

        itr_x += 1
    #-------------------------------------------------------------


    #-----------------------Plot the Results-----------------------
    vals_exc = means_vector[0,:]
    std_exc  = std_vector[0,:]
                                                
    vals_inh = means_vector[1,:]
    std_inh = std_vector[1,:]
                                                
    vals_void = means_vector[2,:]
    std_void = std_vector[2,:]

    fig, axs = plt.subplots(nrows=1, ncols=1)
    ax = axs
        
    
    ax.errorbar(x_axis_values,vals_exc,std_exc,color='r',label='Excitatory')
    ax.errorbar(x_axis_values,vals_inh,std_inh,color='b',label='Inhibitory');
    ax.errorbar(x_axis_values,vals_void,std_void,color='g',label='Void');
            
    plt.xlabel(x_label, fontsize=16)    
    if not y_label:
        plt.ylabel('Average of beliefs', fontsize=16)
    else:
        plt.ylabel(y_label, fontsize=16)
    
    plt.legend(loc='lower left')
    ax.set_title('Average belief qualities')
    plt.show();
    #-------------------------------------------------------------

    #------------------Export Plots to Plotly---------------------
    if plotly_flag:
        plot_legends = ['Excitatory','Inhibitory','Void']
        plot_colors = ['#F62817','#1569C7','#4CC552']
        error_bar_colors = ['#F75D59','#368BC1','#54C571']
        
        x_array = np.vstack([x_axis_values,x_axis_values,x_axis_values])
        y_array = np.vstack([vals_exc,vals_inh,vals_void])
        error_array = np.vstack([std_exc,std_inh,std_void])                
        no_plots = 3
                
        plot_title = 'Average belief qualities'
        plot_url = export_to_plotly(x_array,y_array,no_plots,plot_legends,'line',plot_colors,'t(s)','Average of beliefs',plot_title,error_array,error_bar_colors)
    #-------------------------------------------------------------

#=================================================================


#===========================Plot the ROC Curves===================

#=================================================================
     
#------------------------------------ROC Curve--------------------------------
if 'R' in plot_vars:
    n,m = W.shape
    m = n
    W_inferred = np.zeros([n,m])
    
    W_infer = np.zeros([len(file_name_ending_list22),n])
    
    if 1:
        W_inferred2 = np.zeros([n,len(neuron_range)])
        for file_name_ending in file_name_ending_list22:
            file_name = file_name_base_results + "/Inferred_Graphs/W_Pll_%s_n_%s_%s.txt" %(file_name_ending,str(neuron_range[0]),str(neuron_range[-1]))
            W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
            #W_inferred = W_inferred + np.multiply(W_read[0:n,:],(W_read[0:n,:]>0).astype(int))
            W_inferred2 = W_inferred2 + W_read[0:n,:]
            itr_i = itr_i + 1    
        
        W_inferred[:,neuron_range] = W_inferred2
    else:
        itr_n = 0
        for n_ind in neuron_range:
            
            #~~~~~~~~~~~~~~~~~~~~Read the Inferred Weights~~~~~~~~~~~~~~~~~~~~
            itr_i = 0
            for file_name_ending in file_name_ending_list2:
                
                file_name = file_name_base_results + "/Inferred_Graphs/W_Pll_%s_%s.txt" %(file_name_ending,str(n_ind))
                W_read = np.genfromtxt(file_name, dtype=None, delimiter='\t')
                W_infer[itr_i,:] = W_read[0:n]
                
                itr_i = itr_i + 1
                
            if itr_i > 1:
                W_inferred[0:min(m,len(W_read)),itr_n] = W_infer[0:itr_i,:].mean(axis = 0)
            else:
                W_inferred[0:min(m,len(W_read)),itr_n] = W_infer
                
            #W_inferred[0:min(m,len(W_read)),n_ind] = W_read[0:min(m,len(W_read))]
            W_inferred[0:min(m,len(W_read)),itr_n] = W_inferred[0:min(m,len(W_read)),itr_n] - W_inferred[0:min(m,len(W_read)),itr_n].min()
            itr_n = itr_n + 1
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        
    #W_inferred = W_inferred/np.abs(W_inferred).max()
    val_range = range(0,500)
    val_range = np.array(val_range)
    val_range = val_range/500.0
    
    true_pos = np.zeros([len(val_range)])
    false_pos = np.zeros([len(val_range)])
    
    itr_n = 0
    for n_ind in neuron_range:
        itr_V1 = 0
        no_edges = 0
        no_edges_0 = 0
        
        
        W_temp = (W_inferred[:,itr_n])
        W_temp = W_temp - W_temp.min()
        W_temp = np.multiply(W_temp,(W_temp>0).astype(int))
        W_temp = W_temp/(0.00001+ np.abs(W_temp).max())
        W_inferred[:,itr_n] = W_temp
        #W_temp = W_temp[:-1]    
        
        itr_n = itr_n + 1    
    
    W_s = W[0:n,neuron_range]
    for thresh in val_range:        
        #~~~~~~~~Calcualte True Positive and False Positive Rates~~~~~~~~~
        W_ter = (W_inferred[0:n,neuron_range]>=thresh).astype(int)
            
        true_pos[itr_V1] = true_pos[itr_V1] + sum(sum(np.multiply((W_ter>0).astype(int),(W_s>0).astype(int))))/float(sum(sum(W_s>0)))
        false_pos[itr_V1] = false_pos[itr_V1] + sum(sum(np.multiply((W_ter>0).astype(int),(W_s==0).astype(int))))/float(sum(sum(W_s==0)))
        #no_edges = no_edges + sum(W[:,n_ind]>0)
        #no_edges_0 = no_edges_0 + sum(W[:,n_ind]==0)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        itr_V1 = itr_V1 + 1
        
    #true_pos = true_pos/float(len(neuron_range))
    #false_pos = false_pos/float(len(neuron_range))
            
    plt.plot(false_pos,true_pos,linewidth=3,label='Only positive weights');plt.plot(val_range,val_range,'r',linewidth=2);
    #plt.plot(false_pos2,true_pos2,linewidth=3,label='All weights')
    
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)    
    plt.title('ROC curve for the dataset with 60s of data')
    plt.legend(loc='lower right')
    plt.show()
    
    
    TFR = np.divide(true_pos-false_pos,false_pos+true_pos)
    TFS = false_pos+true_pos
    plt.plot(TFS,TFR,linewidth=3);
    
    plt.xlabel('True False Sum (TFS)', fontsize=16)
    plt.ylabel('True False Ration (TFR)', fontsize=16)    
    plt.title('PPC curve for the dataset with 60s of data')
    
    plt.show()

#------------------------------------------------------------------------------

#==============================================================================


#=================================PLOT THE RESULTS=============================
            
#-----------------------Plot the Scatter of Beliefs----------------------------
if 0:    
    if whiten_flag:
        n,m = W_inferred.shape
        W_inferred = W_inferred + np.random.rand(n,m)/100000
        W_inferred  = whiten(W_inferred)
         
    if 'S' in plot_type:
        plt.title('Scatter plot of belief')
        plt.scatter(np.sign(W.ravel()),W_inferred.ravel())
        plt.xlabel('G (actual)', fontsize=16)
        plt.ylabel('W (inferred)', fontsize=16)
        plt.show()
    else:
        W_inferred = np.array([0])
#------------------------------------------------------------------------------

#-------------------------------Save the Results-------------------------------
if 'B' in plot_type:
    save_plot_results(Var1_range,mean_exc,std_exc,mean_inh,std_inh,mean_void,
                          std_void,0,0,file_name_base_results,
                          file_name_ending,0,W_inferred,W)
        
        
if 'P' in plot_type:
    
    #???? Fix these

    std_Prec_exc = 0*Prec_exc
    std_Prec_inh = 0*Prec_inh
    std_Prec_void = 0*Prec_void
    std_Rec_exc = 0*Rec_exc
    std_Rec_inh = 0*Rec_inh
    std_Rec_void = 0*Rec_void
    
    save_precision_recall_results(Var1_range,file_name_base_results,file_name_ending2,adj_fact_exc,adj_fact_inh,ternary_mode,
                                      Prec_exc,std_Prec_exc,Prec_inh,std_Prec_inh,Prec_void,
                                      std_Prec_void,Rec_exc,std_Rec_exc,Rec_inh,std_Rec_inh,Rec_void,std_Rec_void)


if 'S' in plot_type:
    spent_ram = spent_ram/float(1e6)            # Transform to GB
    spent_cpu = spent_cpu/3600.                 # Transform to hours
    temp = np.vstack([np.array(Var1_range).T,(spent_cpu).T,spent_ram.T])
    file_name = file_name_base_results + "/Plot_Results/CPU_RAM_%s.txt" %file_name_ending
    np.savetxt(file_name,temp.T,'%3.5f',delimiter='\t',newline='\n')
#------------------------------------------------------------------------------

#==============================================================================


#=================SAVE RESULTS FOR WEB DEMO IF NECESSARY=======================
pdb.set_trace()
if 0:
    save_web_demo(W,W_inferred,file_name_base_results,file_name_ending)
#==============================================================================


if 0:
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

    
no_hidden_neurons_list = np.array(no_hidden_neurons_list)
no_hidden_neurons_list_str = no_hidden_neurons_list.astype('str')
temp_ending = file_name_ending + '_' + '_'.join(no_hidden_neurons_list_str)
file_name = file_name_base_results + "/Plot_Results/Prec_Reca_All_Effect_hidden_%s.txt" %(temp_ending)

hidden_visible_ration = np.divide(no_hidden_neurons_list,(n-no_hidden_neurons_list).astype(float))
temp = np.vstack([prec_exc_h,prec_inh_h,prec_void_h,rec_exc_h,rec_inh_h,rec_void_h])
temp = temp.T
temp = np.hstack([np.reshape(hidden_visible_ration,[len(hidden_visible_ration),1]),temp])

np.savetxt(file_name,temp,'%f',delimiter='\t',newline='\n')
