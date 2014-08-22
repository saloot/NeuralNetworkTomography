#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.2
frac_input_neurons_default = 0.4
no_cascades_default = 8000
ensemble_size_default = 10
binary_mode_default = 4
delay_max_default = 1.0
inference_method = 0
#==============================================================================


#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import time
import numpy as np
import sys,getopt,os
from time import time
import matplotlib.pyplot as plt

#os.chdir('C:\Python27')
#==============================================================================


#================================INITIALIZATIONS===============================

#------------Set the Default Values if Variables are not Defines---------------
if 'n_exc' not in locals():
    n_exc = n_exc_default
    print('ATTENTION: The default value of %s for n_exc is considered.\n' %str(n_exc))

if 'n_inh' not in locals():
    n_inh = n_inh_default
    print('ATTENTION: The default value of %s for n_inh is considered.\n' %str(n_inh))

if 'connection_prob' not in locals():    
    connection_prob = connection_prob_default
    print('ATTENTION: The default value of %s for connection_prob is considered.\n' %str(connection_prob))

if 'frac_input_neurons' not in locals():
    frac_input_neurons = frac_input_neurons_default
    print('ATTENTION: The default value of %s for frac_input_neurons is considered.\n' %str(frac_input_neurons))

if 'no_cascades' not in locals():        
    no_cascades = no_cascades_default
    print('ATTENTION: The default value of %s for no_cascades is considered.\n' %str(no_cascades))

if 'ensemble_size' not in locals():            
    ensemble_size = ensemble_size_default
    print('ATTENTION: The default value of %s for ensemble_size is considered.\n' %str(ensemble_size))

if 'binary_mode' not in locals():
    binary_mode = binary_mode_default;
    print('ATTENTION: The default value of %s for binary_mode is considered.\n' %str(binary_mode))
    
if 'delay_max' not in locals():
    delay_max = delay_max_default;
    print('ATTENTION: The default value of %s for delay_max is considered.\n' %str(delay_max))    
#------------------------------------------------------------------------------



#--------------------------Initialize Other Variables--------------------------
T_range = range(50, no_cascades, 250)                   # The range of sample sizes considered to investigate the effect of sample size on the performance
q_range = [0.2,0.3,0.4]                                 # The range of stimulated fraction of neurons considered in simulations
a_range = np.arange(0.0,2,0.125)

considered_var = 'T'                                    # The variable against which we plot the performance
network_type = 'F'                                      # The type of the network considered in plotting the results, 'F' for feedforward and 'R' for recurrent

if (considered_var == 'T'):
    var_range = T_range
elif (considered_var == 'q'):
    var_range = q_range                                 # The variable against which we plot the performance
elif (considered_var == 'a'):
    var_range = a_range 
        

n = n_exc + n_inh                       # Total number of neurons in the output layer

theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_inh = connection_prob * (float(n_inh)/float(n))
p_exc = connection_prob * (float(n_exc)/float(n))


if (network_type == 'R'):
    file_name_base_results = "./Results/Recurrent"                  #The folder that stores the resutls
    file_name_base_plot = "./Results/Recurrent/Plot_Results"        #The folder to store resutls
    name_base = 'Delayed_'
    
elif (network_type == 'F'):
    file_name_base_results = "./Results/FeedForward"                #The folder that stores the resutls
    file_name_base_plot = "./Results/FeedForward/Plot_Results"      #The folder to store resutls
    name_base = 'FF_n_to_1_'    

if (inference_method == 0):
    name_prefix =  ''
else:
    name_prefix = 'Hebbian_'
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if Necessary---------------
if not os.path.isdir(file_name_base_plot):
    os.makedirs(file_name_base_plot)
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
det_Prec_exc = np.zeros([ensemble_size,len(var_range)])                        # Detailed precision of the algorithm (per ensemble) for excitatory connections
det_Prec_inh = np.zeros([ensemble_size,len(var_range)])                        # Detailed precision of the algorithm (per ensemble) for inhibitory connections
det_Prec_zero  = np.zeros([ensemble_size,len(var_range)])                      # Detailed precision of the algorithm (per ensemble) for "void" connections

Prec_exc = np.zeros([len(var_range)])                                          # Precision of the algorithm for excitatory connections
Prec_inh = np.zeros([len(var_range)])                                          # Precision of the algorithm for inhibitory connections
Prec_zero  = np.zeros([len(var_range)])                                        # Precision of the algorithm for "void" connections
Prec_total  = np.zeros([len(var_range)])                                       # Precision of the algorithm averaged over all connections

std_Prec_exc = np.zeros([len(var_range)])                                      # Standard Deviation of precision of the algorithm for excitatory connections
std_Prec_inh = np.zeros([len(var_range)])                                      # Standard Deviation of precision of the algorithm for inhibitory connections
std_Prec_zero  = np.zeros([len(var_range)])                                    # Standard Deviation of precision of the algorithm for "void" connections
std_Prec_total  = np.zeros([len(var_range)])                                   # Standard Deviation of precision of the algorithm averaged over all connections

det_Rec_exc = np.zeros([ensemble_size,len(var_range)])                         # Detailed recall of the algorithm for (per ensemble) excitatory connections
det_Rec_inh = np.zeros([ensemble_size,len(var_range)])                         # Detailed recall of the algorithm for (per ensemble) inhibitory connections
det_Rec_zero  = np.zeros([ensemble_size,len(var_range)])                       # Detailed recall of the algorithm for (per ensemble) "void" connections

Rec_exc = np.zeros([len(var_range)])                                           # Recall of the algorithm for excitatory connections
Rec_inh = np.zeros([len(var_range)])                                           # Recall of the algorithm for inhibitory connections
Rec_zero  = np.zeros([len(var_range)])                                         # Recall of the algorithm for "void" connections
Rec_total= np.zeros([len(var_range)])                                          # Recall of the algorithm averaged over all connections

std_Rec_exc = np.zeros([len(var_range)])                                       # Standard Deviation of recall of the algorithm for excitatory connections
std_Rec_inh = np.zeros([len(var_range)])                                       # Standard Deviation of recall of the algorithm for excitatory connections
std_Rec_zero  = np.zeros([len(var_range)])                                     # Standard Deviation of recall of the algorithm for excitatory connections
std_Rec_total= np.zeros([len(var_range)])                                      # Standard Deviation of recall of the algorithm averaged over all connections

#------------------------------------------------------------------------------

#==============================================================================



#=============================READ THE RESULTS=================================
itr = 0
adj_fact_exc = 1.875 #1 #1.25 #1.75 
adj_fact_inh = 1.875 #1 #1.25 #1.75 
for var in var_range:
    if (considered_var == 'T'):
        T = var
    elif (considered_var == 'q'):        
        frac_input_neurons = var
        T = 7800
    elif (considered_var == 'a'):        
        adj_fact_exc = var
        adj_fact_inh = var
        T = 6800    
        
    for ensemble_count in range(0,ensemble_size):
        
        #----------------------Construct Prpoper File Names------------------------
        file_name_ending = name_base + "n_exc_%s" %str(int(n_exc))
        file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
        file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
        file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
        #file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
        file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
        file_name_ending = file_name_ending + "_%s" %str(adj_fact_exc)
        file_name_ending = file_name_ending + "_%s" %str(adj_fact_inh)
        #file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
        file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
        file_name_ending = file_name_ending + "_T_%s" %str(T)
        file_name_ending = file_name_ending + "_B_%s" %str(binary_mode)
        
        #file_name_ending = file_name_ending + "_I_%s" %str(0)
        #--------------------------------------------------------------------------
    
        #----------------------------Read the Precisions---------------------------
        file_name = file_name_base_results + "/Accuracies/"
        file_name = file_name + name_prefix + "Prec_%s.txt" %file_name_ending
        Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        s = Acc.shape
        if (len(s) > 1):
            temp = (sum(Acc,axis=0))/float(s[0])
        else:
            temp = Acc
        
        det_Prec_exc[ensemble_count,itr] = det_Prec_exc[ensemble_count,itr] + temp[0]
        det_Prec_inh[ensemble_count,itr] = det_Prec_inh[ensemble_count,itr] + temp[1]
        det_Prec_zero[ensemble_count,itr] = det_Prec_zero[ensemble_count,itr] + temp[2]
        #--------------------------------------------------------------------------
                
        #------------------------------Read the Recall-----------------------------
        file_name = file_name_base_results + "/Accuracies/"
        file_name = file_name + name_prefix + "Rec_%s.txt" %file_name_ending
        Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        s = Acc.shape
        if (len(s) > 1):
            temp = (sum(Acc,axis=0))/float(s[0])
        else:
            temp = Acc
        
        det_Rec_exc[ensemble_count,itr] = det_Rec_exc[ensemble_count,itr] + temp[0]
        det_Rec_inh[ensemble_count,itr] = det_Rec_inh[ensemble_count,itr] + temp[1]
        det_Rec_zero[ensemble_count,itr] = det_Rec_zero[ensemble_count,itr] + temp[2]
        #--------------------------------------------------------------------------
            

        
    itr = itr + 1
#==============================================================================


#===================CALCULATE THE MEAN AND VAR OF RESULTS======================

#-----------------------------Calculating the Mean-----------------------------
Prec_exc = det_Prec_exc.mean(axis=0)
Prec_inh = det_Prec_inh.mean(axis=0)
Prec_zero = det_Prec_zero.mean(axis=0)
Rec_exc = det_Rec_exc.mean(axis=0)
Rec_inh = det_Rec_inh.mean(axis=0)
Rec_zero = det_Rec_zero.mean(axis=0)

Prec_total = p_exc*Prec_exc+p_inh*Prec_inh+(1-connection_prob)*Prec_zero
Rec_total = p_exc * Rec_exc+p_inh*Rec_inh+(1-connection_prob)*Rec_zero

det_Prec_total = p_exc*det_Prec_exc+p_inh*det_Prec_inh+(1-connection_prob)*det_Prec_zero
det_Rec_total = p_exc * det_Rec_exc+p_inh*det_Rec_inh+(1-connection_prob)*det_Rec_zero
#------------------------------------------------------------------------------


#---------------------Calculating the Standard Deviation-----------------------
std_Prec_exc = det_Prec_exc.std(axis=0)
std_Prec_inh = det_Prec_inh.std(axis=0)
std_Prec_zero = det_Prec_zero.std(axis=0)
std_Prec_total = det_Prec_total.std(axis=0)

std_Rec_exc = det_Rec_exc.std(axis=0)
std_Rec_inh = det_Rec_inh.std(axis=0)
std_Rec_zero = det_Rec_zero.std(axis=0)
std_Rec_total = det_Rec_total.std(axis=0)
#------------------------------------------------------------------------------
#==============================================================================



#==============================PLOT THE RESULTS================================
plot(var_range,Prec_exc,'r')
plot(var_range,Prec_inh,'b')
plot(var_range,Prec_zero,'g')

plot(var_range,Rec_exc,'r--')
plot(var_range,Rec_inh,'b--')
plot(var_range,Rec_zero,'g--')
#==============================================================================


#=========================WRITE THE RESULTS TO THE FILE=========================

#--------------------------Construct Prpoper File Names-------------------------
if (considered_var == 'T'):
    file_name_ending = name_base + name_prefix + "Effect_T_n_exc_%s" %str(int(n_exc))
elif (considered_var == 'q'):        
    file_name_ending = name_base + name_prefix + "Effect_q_n_exc_%s" %str(int(n_exc))
elif (considered_var == 'a'):        
    file_name_ending = name_base + name_prefix + "Effect_a_n_exc_%s" %str(int(n_exc))
    
file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
if (considered_var != 'q'):
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
#file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
file_name_ending = file_name_ending + "_%s" %str(ensemble_size)
if (considered_var != 'T'): 
    file_name_ending = file_name_ending + "_%s" %str(T)
if (considered_var != 'a'): 
    file_name_ending = file_name_ending + "_%s" %str(adj_fact_exc)
    file_name_ending = file_name_ending + "_%s" %str(adj_fact_inh)

file_name_ending = file_name_ending + "_B_%s" %str(binary_mode)
#-------------------------------------------------------------------------------

#-----------------------Write the Results to the File---------------------------
temp = np.vstack([var_range,Prec_exc,std_Prec_exc])
file_name = file_name_base_plot + "/Prec_exc_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Prec_inh,std_Prec_inh])
file_name = file_name_base_plot + "/Prec_inh_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Prec_zero,std_Prec_zero])
file_name = file_name_base_plot + "/Prec_zero_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Prec_total,std_Prec_total])
file_name = file_name_base_plot + "/Prec_total_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Rec_exc,std_Rec_exc])
file_name = file_name_base_plot + "/Reca_exc_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Rec_inh,std_Rec_inh])
file_name = file_name_base_plot + "/Reca_inh_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Rec_zero,std_Rec_zero])
file_name = file_name_base_plot + "/Reca_zero_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Rec_total,std_Rec_total])
file_name = file_name_base_plot + "/Reca_total_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([Rec_exc,Prec_exc,std_Rec_exc,std_Prec_exc])
file_name = file_name_base_plot + "/Reca_vs_Prec_exc_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')

temp = np.vstack([Rec_inh,Prec_inh,std_Rec_inh,std_Prec_inh])
file_name = file_name_base_plot + "/Reca_vs_Prec_inh_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')

temp = np.vstack([Rec_zero,Prec_zero,std_Rec_zero,std_Prec_zero])
file_name = file_name_base_plot + "/Reca_vs_Prec_zero_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')

temp = np.vstack([Rec_total,Prec_total,std_Rec_total,std_Prec_total])
file_name = file_name_base_plot + "/Reca_vs_Prec_total_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%1.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,np.multiply(Prec_exc,Rec_exc)])
file_name = file_name_base_plot + "/Prec_mult_Rec_exc_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,np.multiply(Prec_inh,Rec_inh)])
file_name = file_name_base_plot + "/Prec_mult_Rec_inh_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------



#==============================================================================
