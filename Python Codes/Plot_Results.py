#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.15
frac_input_neurons_default = 0.4
no_cascades_default = 8000
ensemble_size_default = 10
binary_mode_default = 2
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
#------------------------------------------------------------------------------



#--------------------------Initialize Other Variables--------------------------
T_range = range(50, no_cascades, 50)                    # The range of sample sizes considered to investigate the effect of sample size on the performance
q_range = [0.2,0.3,0.4]                                 # The range of stimulated fraction of neurons considered in simulations
var_range = q_range                                     # The variable against which we plot the performance

n = n_exc + n_inh                       # Total number of neurons in the output layer

theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_inh = connection_prob * (float(n_inh)/float(n))
p_exc = connection_prob * (float(n_exc)/float(n))

file_name_base_results = "./Results/Recurrent"       #The folder that stores the resutls
file_name_base_plot = "./Results/Recurrent/Plot_Results"       #The folder to store resutls =
name_base = 'FF_n_1_'
name_base = 'Delayed_'

if (inference_method == 0):
    name_prefix = ''
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

simulation_time = np.zeros([ensemble_size,len(var_range)])                  # The running time of the inference algorithm per ensemble
#------------------------------------------------------------------------------

#==============================================================================



#=============================READ THE RESULTS=================================
itr = 0

for frac_input_neurons in q_range:
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
        file_name_ending = file_name_ending + "_%s" %str(T)
        file_name_ending = file_name_ending + "_%s" %str(binary_mode)
        file_name_ending = file_name_ending + "_I_%s" %str(0)
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
            
        #--------------------------Read the Running Time---------------------------
        file_name = file_name_base_results + "/RunningTimes/T_%s.txt" %file_name_ending
        T_run = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
        simulation_time[ensemble_count,itr] = sum(T_run)/len(T_run)
        #--------------------------------------------------------------------------
        
    itr = itr + 1
#==============================================================================


#===================CALCULATE THE MEAN AND VAR OF RESULTS======================

#-----------------------------Calculating the Mean-----------------------------
simulation_time_tot = simulation_time.mean(axis=0)
Prec_exc = det_Prec_exc.mean(axis=0)
Prec_inh = det_Prec_inh.mean(axis=0)
Prec_zero = det_Prec_zero.mean(axis=0)
Rec_exc = det_Rec_exc.mean(axis=0)
Rec_inh = det_Rec_inh.mean(axis=0)
Rec_zero = det_Rec_zero.mean(axis=0)

Prec_exc_hebb = det_Prec_exc_hebb.mean(axis=0)
Prec_inh_hebb = det_Prec_inh_hebb.mean(axis=0)
Prec_zero_hebb = det_Prec_zero_hebb.mean(axis=0)
Rec_exc_hebb = det_Rec_exc_hebb.mean(axis=0)
Rec_inh_hebb = det_Rec_inh_hebb.mean(axis=0)
Rec_zero_hebb = det_Rec_zero_hebb.mean(axis=0)


Prec_total_hebb = p_exc*Prec_exc_hebb+p_inh*Prec_inh_hebb+(1-connection_prob)*Prec_zero_hebb
Rec_total_hebb = p_exc * Rec_exc_hebb+p_inh*Rec_inh_hebb+(1-connection_prob)*Rec_zero_hebb

Prec_total = p_exc*Prec_exc+p_inh*Prec_inh+(1-connection_prob)*Prec_zero
Rec_total = p_exc * Rec_exc+p_inh*Rec_inh+(1-connection_prob)*Rec_zero
#------------------------------------------------------------------------------


#---------------------Calculating the Standard Deviation-----------------------
std_Prec_exc = det_Prec_exc.std(axis=0)
std_Prec_inh = det_Prec_inh.std(axis=0)
std_Prec_zero = det_Prec_zero.std(axis=0)
std_Prec_total = Prec_total.std(axis=0)

std_Rec_exc = Rec_exc.std(axis=0)
std_Rec_inh = Rec_inh.std(axis=0)
std_Rec_zero = Rec_zero.std(axis=0)
std_Rec_total = Rec_total.std(axis=0)

std_Prec_exc_hebb = Prec_exc_hebb.std(axis=0)
std_Prec_inh_hebb = Prec_inh_hebb.std(axis=0)
std_Prec_zero_hebb = Prec_zero_hebb.std(axis=0)
std_Prec_total_hebb = Prec_total_hebb.std(axis=0)

std_Rec_exc_hebb = Rec_exc_hebb.std(axis=0)
std_Rec_inh_hebb = Rec_inh_hebb.std(axis=0)
std_Rec_zero_hebb = Rec_zero_hebb.std(axis=0)
std_Rec_total_hebb = Rec_total_hebb.std(axis=0)

std_simulation_time = simulation_time.std(axis=0)
#------------------------------------------------------------------------------
#==============================================================================



#==============================PLOT THE RESULTS================================
plot(var_range,Prec_exc,'b')
plot(var_range,Prec_inh,'r')
plot(var_range,Prec_zero,'g')

plot(var_range,Rec_exc,'b--')
plot(var_range,Rec_inh,'r--')
plot(var_range,Rec_zero,'g--')
#==============================================================================


#=========================WRITE THE RESULTS TO THE FILE=========================

#--------------------------Construct Prpoper File Names-------------------------
file_name_ending = name_base + name_prefix + "Effect_q_n_exc_%s" %str(int(n_exc))
file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
#file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
#file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
file_name_ending = file_name_ending + "_%s" %str(ensemble_size)
file_name_ending = file_name_ending + "_%s" %str(T)
file_name_ending = file_name_ending + "_%s" %str(binary_mode)
#-------------------------------------------------------------------------------

#-----------------------Write the Results to the File---------------------------
temp = np.vstack([var_range,Prec_exc,std_Prec_exc])
file_name = file_name_base_plot + "/Prec_exc_vs_T_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Prec_inh,std_Prec_inh])
file_name = file_name_base_plot + "/Prec_inh_vs_T_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Prec_zero,std_Prec_zero])
file_name = file_name_base_plot + "/Prec_zero_vs_T_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Prec_total,std_Prec_total])
file_name = file_name_base_plot + "/Prec_total_vs_T_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Rec_exc,std_Rec_exc])
file_name = file_name_base_plot + "/Reca_exc_vs_T_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Rec_inh,std_Rec_inh])
file_name = file_name_base_plot + "/Reca_inh_vs_T_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Rec_zero,std_Rec_zero])
file_name = file_name_base_plot + "/Reca_zero_vs_T_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%4.3f',delimiter='\t',newline='\n')

temp = np.vstack([var_range,Rec_total,std_Rec_total])
file_name = file_name_base_plot + "/Reca_total_vs_T_%s.txt" %file_name_ending
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
#-------------------------------------------------------------------------------

#-------------------Write the Running Times on the File-------------------------
temp = np.vstack([var_range,simulation_time_tot,std_simulation_time])
file_name = file_name_base_plot + "/Running_time_%s.txt" %file_name_ending
np.savetxt(file_name,temp.T,'%5.2f',delimiter='\t',newline='\n')
#-------------------------------------------------------------------------------

#==============================================================================
