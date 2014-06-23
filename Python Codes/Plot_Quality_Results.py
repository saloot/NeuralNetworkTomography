#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.15
frac_input_neurons_default = 0.4
no_cascades_default = 8000
ensemble_size_default = 10
binary_mode_default = 2
delay_max_default = 1.0
inference_method_default = 0
#==============================================================================


#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import time
import numpy as np
import sys,getopt,os
from time import time
import matplotlib.pyplot as plt

os.chdir('C:\Python27')
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
    
if 'inference_method' not in locals():
    inference_method = inference_method_default;
    print('ATTENTION: The default value of %s for inference_method is considered.\n' %str(inference_method))
#------------------------------------------------------------------------------



#--------------------------Initialize Other Variables--------------------------
T_range = range(10, no_cascades, 200)                    # The range of sample sizes considered to investigate the effect of sample size on the performance
q_range = [0.2,0.3,0.4]                                 # The range of stimulated fraction of neurons considered in simulations

considered_var = 'T'                                    # The variable against which we plot the performance
network_type = 'R'                                      # The type of the network considered in plotting the results, 'F' for feedforward and 'R' for recurrent

if (considered_var == 'T'):
    var_range = T_range
elif (considered_var == 'q'):
    var_range = q_range

n = n_exc + n_inh                       # Total number of neurons in the output layer

theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_inh = connection_prob * (float(n_inh)/float(n))
p_exc = connection_prob * (float(n_exc)/float(n))

if (network_type == 'R'):
    file_name_base_results = "./Results/Recurrent"       #The folder that stores the resutls
    file_name_base_plot = "./Results/Recurrent/Plot_Results"       #The folder to store resutls =
    name_base = 'Delayed_'
    name_prefix = "BQ_"
elif (network_type == 'F'):
    file_name_base_results = "./Results/FeedForward"       #The folder that stores the resutls
    file_name_base_plot = "./Results/FeedForward/Plot_Results"       #The folder to store resutls =
    name_base = 'FF_n_1_'
    name_prefix = "BQ_FF_"
    
if (inference_method == 0):
    name_prefix = name_prefix + ''
else:
    name_prefix = name_prefix + 'Hebbian_'
#------------------------------------------------------------------------------

#------------------Create the Necessary Directories if Necessary---------------
if not os.path.isdir(file_name_base_plot):
    os.makedirs(file_name_base_plot)
#------------------------------------------------------------------------------

#---------------------Initialize Simulation Variables--------------------------
B_exc_min_Det = np.zeros([ensemble_size,len(var_range)])                        # Minimum beliefs (per ensemble) for excitatory connections
B_inh_max_Det = np.zeros([ensemble_size,len(var_range)])                        # Maximum beliefs (per ensemble) (per ensemble) for inhibitory connections
B_zero_min_Det  = np.zeros([ensemble_size,len(var_range)])                      # Minimum beliefs (per ensemble) (per ensemble) for "void" connections
B_zero_max_Det  = np.zeros([ensemble_size,len(var_range)])                      # Maximum beliefs (per ensemble) (per ensemble) for "void" connections

B_exc_min = np.zeros([len(var_range)])                                          # Average of minimum beliefs for excitatory connections
B_inh_max = np.zeros([len(var_range)])                                          # Average of maximum beliefs for inhibitory connections
B_zero_min  = np.zeros([len(var_range)])                                        # Average of minimum beliefs for "void" connections
B_zero_max  = np.zeros([len(var_range)])                                        # Average of maximum beliefs for "void" connections

B_exc_min_std = np.zeros([len(var_range)])                                      # Standard deviations of minimum beliefs for excitatory connections
B_inh_max_std = np.zeros([len(var_range)])                                      # Standard deviations of maximum beliefs for inhibitory connections
B_zero_min_std  = np.zeros([len(var_range)])                                    # Standard deviations of minimum beliefs for "void" connections
B_zero_max_std  = np.zeros([len(var_range)])                                    # Standard deviations of maximum beliefs for "void" connections

simulation_time = np.zeros([ensemble_size,len(var_range)])                  # The running time of the inference algorithm per ensemble
#------------------------------------------------------------------------------

#==============================================================================



#=============================READ THE RESULTS=================================
itr = 0

for var in var_range:
    if (considered_var == 'T'):
        T = var
    elif (considered_var == 'q'):        
        frac_input_neurons = var
        T = 8100

    for ensemble_count in range(0,ensemble_size):
    
        #----------------------Construct Prpoper File Names------------------------
        file_name_ending = name_base + "n_exc_%s" %str(int(n_exc))
        file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
        file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
        file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
        #file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
        file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
        file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
        file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
        file_name_ending2 = file_name_ending + "_%s" %str(T)
        #--------------------------------------------------------------------------
                
        #------------------Read the Standard Deviation of Beliefs------------------        
        file_name = file_name_base_results + "/BeliefQuality/"
        file_name = file_name + name_prefix + "Max_Min_Cascades_%s.txt" %file_name_ending2
        Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        s = Acc.shape
        if (len(s) > 1):
            temp = (sum(Acc,axis=0))/float(s[0])
        else:
            temp = Acc
        
        B_exc_min_Det[ensemble_count,itr] = B_exc_min_Det[ensemble_count,itr] + temp[0]
        B_zero_max_Det[ensemble_count,itr] = B_zero_max_Det[ensemble_count,itr] + temp[1]
        B_zero_min_Det[ensemble_count,itr] = B_zero_min_Det[ensemble_count,itr] + temp[2]
        B_inh_max_Det[ensemble_count,itr] = B_inh_max_Det[ensemble_count,itr] + temp[3]
        #--------------------------------------------------------------------------
            
        #--------------------------Read the Running Time---------------------------
        file_name_ending3 = file_name_ending + "_%s" %str(binary_mode)        
        file_name_ending3 = file_name_ending + "_%s" %str(T)
        if (network_type=='R'):
            file_name = file_name_base_results + "/RunningTimes/T_Cascades_%s.txt" %file_name_ending3
        else:
            file_name = file_name_base_results + "/RunningTimes/T_%s.txt" %file_name_ending3
        T_run = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
        simulation_time[ensemble_count,itr] = sum(T_run)#/len(T_run)
        #--------------------------------------------------------------------------
        
    itr = itr + 1
#==============================================================================


#===================CALCULATE THE MEAN AND VAR OF RESULTS======================

#-----------------------------Calculating the Mean-----------------------------
B_exc_min = B_exc_min_Det.mean(axis=0)
B_inh_max = B_inh_max_Det.mean(axis=0)
B_zero_min = B_zero_min_Det.mean(axis=0)
B_zero_max = B_zero_max_Det.mean(axis=0)

simulation_time_tot = simulation_time.mean(axis=0)
#------------------------------------------------------------------------------


#---------------------Calculating the Standard Deviation-----------------------
B_exc_min_std = B_exc_min_Det.std(axis=0)
B_inh_max_std = B_inh_max_Det.std(axis=0)
B_zero_min_std = B_zero_min_Det.std(axis=0)
B_zero_max_std = B_zero_max_Det.std(axis=0)

std_simulation_time = simulation_time.std(axis=0)
#------------------------------------------------------------------------------
#==============================================================================



#==============================PLOT THE RESULTS================================
#plt.errorbar(var_range,B_exc_min,B_exc_min_std,color='r')
#plt.errorbar(var_range,B_zero_max,B_zero_max_std,color='g')
#plt.errorbar(var_range,B_zero_min,B_zero_min_std,color='k')
#plt.errorbar(var_range,B_inh_max,B_inh_max_std,color='b')

plt.plot(var_range,B_exc_min,'r')
plt.plot(var_range,B_zero_max,'g')
plt.plot(var_range,B_zero_min,'g--')
plt.plot(var_range,B_inh_max,'b')
#==============================================================================


#=========================WRITE THE RESULTS TO THE FILE=========================

#--------------------------Construct Prpoper File Names-------------------------
if (considered_var == 'T'):
    e_ending = name_base + name_prefix + "Effect_T_n_exc_%s" %str(int(n_exc))
elif (considered_var == 'q'):        
    e_ending = name_base + name_prefix + "Effect_q_n_exc_%s" %str(int(n_exc))        

file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
if (considered_var == 'T'):
    file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
#file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
file_name_ending = file_name_ending + "_%s" %str(ensemble_size)
if (considered_var == 'q'):        
    file_name_ending = file_name_ending + "_%s" %str(T)
#-------------------------------------------------------------------------------

#-----------------------Write the Results to the File---------------------------
temp = np.vstack([var_range,B_exc_min,B_exc_min_std])
file_name = file_name_base_plot + "/Quality_exc_min_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%1.4f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,B_inh_max,B_inh_max_std])
file_name = file_name_base_plot + "/Quality_inh_max_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%1.4f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,B_zero_max,B_zero_max_std])
file_name = file_name_base_plot + "/Quality_zero_max_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%1.4f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,B_zero_min,B_zero_min_std])
file_name = file_name_base_plot + "/Quality_zero_min_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%1.4f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------

#-------------------Write the Running Times on the File-------------------------
temp = np.vstack([var_range,simulation_time_tot,std_simulation_time])
file_name = file_name_base_plot + "/Running_time_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%5.2f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------

#==============================================================================
