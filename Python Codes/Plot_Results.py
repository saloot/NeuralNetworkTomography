#=======================DEFAULT VALUES FOR THE VARIABLES=======================
n_exc_default = 160
n_inh_default = 40
connection_prob_default = 0.15
frac_input_neurons_default = 0.2
no_cascades_default = 10000
ensemble_size_default = 5
binary_mode_default = 2
delay_max_default = 1.0                     
#==============================================================================


#=======================IMPORT THE NECESSARY LIBRARIES=========================
from brian import *
import time
import numpy as np
import sys,getopt,os
from time import time
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
os.chdir('C:\Python27')

n = n_exc + n_inh                       # Total number of neurons in the output layer


theta = 10                              # The update threshold of the neurons in the network

input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

p_minus = connection_prob * (float(n_inh)/float(n))
p_plus = connection_prob * (float(n_exc)/float(n))

file_name_base_results = "./Results/Recurrent"       #The folder that stores the resutls
file_name_base_plot = "./Results/Plot_Results"       #The folder to store resutls =
name_base = 'FF_n_1_'
name_base = 'Delayed_'
if not os.path.isdir(file_name_base_plot):
    os.makedirs(file_name_base_plot)
#------------------------------------------------------------------------------

#==============================================================================
range_T = range(100, no_cascades, 500)

range_q = [0.2,0.3,0.4]
range_T = range_q
det_P_C_plus = zeros([ensemble_size,len(range_T)])
det_P_C_minus = zeros([ensemble_size,len(range_T)])
det_P_C_zero  = zeros([ensemble_size,len(range_T)])

det_Rec_plus = zeros([ensemble_size,len(range_T)])
det_Rec_minus = zeros([ensemble_size,len(range_T)])
det_Rec_zero  = zeros([ensemble_size,len(range_T)])

P_C_plus = zeros([len(range_T)])
P_C_minus = zeros([len(range_T)])
P_C_zero  = zeros([len(range_T)])

Rec_plus = zeros([len(range_T)])
Rec_minus = zeros([len(range_T)])
Rec_zero  = zeros([len(range_T)])

var_P_C_plus = zeros([len(range_T)])
var_P_C_minus = zeros([len(range_T)])
var_P_C_zero  = zeros([len(range_T)])

var_Rec_plus = zeros([len(range_T)])
var_Rec_minus = zeros([len(range_T)])
var_Rec_zero  = zeros([len(range_T)])

Prec_total  = zeros([len(range_T)])
Rec_total= zeros([len(range_T)])

var_Prec_total  = zeros([len(range_T)])
var_Rec_total= zeros([len(range_T)])



#------------------For Hebbian------------
det_P_C_plus_hebb = zeros([ensemble_size,len(range_T)])
det_P_C_minus_hebb = zeros([ensemble_size,len(range_T)])
det_P_C_zero_hebb  = zeros([ensemble_size,len(range_T)])

det_Rec_plus_hebb = zeros([ensemble_size,len(range_T)])
det_Rec_minus_hebb = zeros([ensemble_size,len(range_T)])
det_Rec_zero_hebb  = zeros([ensemble_size,len(range_T)])


run_time = np.zeros([ensemble_size,len(range_q)])

P_C_plus_hebb = zeros([len(range_T)])
P_C_minus_hebb = zeros([len(range_T)])
P_C_zero_hebb  = zeros([len(range_T)])

Rec_plus_hebb = zeros([len(range_T)])
Rec_minus_hebb = zeros([len(range_T)])
Rec_zero_hebb  = zeros([len(range_T)])

var_P_C_plus_hebb = zeros([len(range_T)])
var_P_C_minus_hebb = zeros([len(range_T)])
var_P_C_zero_hebb  = zeros([len(range_T)])

var_Rec_plus_hebb = zeros([len(range_T)])
var_Rec_minus_hebb = zeros([len(range_T)])
var_Rec_zero_hebb  = zeros([len(range_T)])

Prec_total_hebb  = zeros([len(range_T)])
Rec_total_hebb = zeros([len(range_T)])

var_Prec_total_hebb   = zeros([len(range_T)])
var_Rec_total_hebb = zeros([len(range_T)])


#P_C_plus_thr = zeros([len(range_T)])
#P_C_minus_thr = zeros([len(range_T)])
#P_C_zero_thr = zeros([len(range_T)])
itr = 0
#for T in range_T:
for frac_input_neurons in range_q:
    T = 8100
    for ensemble_count in range(0,ensemble_size):

    #======================READ THE NETWORK AND SPIKE TIMINGS======================
    
        #----------------------Construct Prpoper File Names------------------------
        file_name_ending = name_base + "n_exc_%s" %str(int(n_exc))
        file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
        file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
        file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
        file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
        file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
        file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
        file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
        file_name_ending = file_name_ending + "_%s" %str(T)
        file_name_ending = file_name_ending + "_%s" %str(binary_mode)
        file_name_ending = file_name_ending + "_I_%s" %str(0)
        #--------------------------------------------------------------------------
    
        #----------------------------Read the Precisions---------------------------
        file_name = file_name_base_results + "/Accuracies/Prec_%s.txt" %file_name_ending
        Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        s = Acc.shape
        if (len(s) > 1):
            temp = (sum(Acc,axis=0))/float(s[0])
        else:
            temp = Acc
        
        det_P_C_plus[ensemble_count,itr] = det_P_C_plus[ensemble_count,itr] + temp[0]
        det_P_C_minus[ensemble_count,itr] = det_P_C_minus[ensemble_count,itr] + temp[1]
        det_P_C_zero[ensemble_count,itr] = det_P_C_zero[ensemble_count,itr] + temp[2]
        #--------------------------------------------------------------------------
                
        #------------------------------Read the Recall-----------------------------
        file_name = file_name_base_results + "/Accuracies/Rec_%s.txt" %file_name_ending
        Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        s = Acc.shape
        if (len(s) > 1):
            temp = (sum(Acc,axis=0))/float(s[0])
        else:
            temp = Acc
        
        det_Rec_plus[ensemble_count,itr] = det_Rec_plus[ensemble_count,itr] + temp[0]
        det_Rec_minus[ensemble_count,itr] = det_Rec_minus[ensemble_count,itr] + temp[1]
        det_Rec_zero[ensemble_count,itr] = det_Rec_zero[ensemble_count,itr] + temp[2]
        #--------------------------------------------------------------------------


        #----------------------------Read the Precisions---------------------------
        file_name = file_name_base_results + "/Accuracies/Hebian_Prec_%s.txt" %file_name_ending
        Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        s = Acc.shape
        if (len(s) > 1):
            temp = (sum(Acc,axis=0))/float(s[0])
        else:
            temp = Acc
        
        det_P_C_plus_hebb[ensemble_count,itr] = det_P_C_plus_hebb[ensemble_count,itr] + temp[0]
        det_P_C_minus_hebb[ensemble_count,itr] = det_P_C_minus_hebb[ensemble_count,itr] + temp[1]
        det_P_C_zero_hebb[ensemble_count,itr] = det_P_C_zero_hebb[ensemble_count,itr] + temp[2]
        #--------------------------------------------------------------------------
                
        #------------------------------Read the Recall-----------------------------
        file_name = file_name_base_results + "/Accuracies/Hebian_Rec_%s.txt" %file_name_ending
        Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        
        s = Acc.shape
        if (len(s) > 1):
            temp = (sum(Acc,axis=0))/float(s[0])
        else:
            temp = Acc
        
        det_Rec_plus_hebb[ensemble_count,itr] = det_Rec_plus_hebb[ensemble_count,itr] + temp[0]
        det_Rec_minus_hebb[ensemble_count,itr] = det_Rec_minus_hebb[ensemble_count,itr] + temp[1]
        det_Rec_zero_hebb[ensemble_count,itr] = det_Rec_zero_hebb[ensemble_count,itr] + temp[2]
        #--------------------------------------------------------------------------

        
        #--------------------------------------------------------------------------
        #Theory
        #        
        # file_name = file_name_base_results + "/Accuracies/Acc_thr_%s.txt" %file_name_ending
        # Acc = np.genfromtxt(file_name, dtype=None, delimiter='\t')
        # s = Acc.shape
        # if (len(s) > 1):
        #    temp = (sum(Acc,axis=0))/float(s[0])
        # else:
        #    temp = Acc
        # P_C_plus_thr[itr] = P_C_plus_thr[itr] + temp[0]
        # P_C_minus_thr[itr] = P_C_minus_thr[itr] + temp[1]
        # P_C_zero_thr[itr] = P_C_zero_thr[itr] + temp[2]

     
    
        
        #--------------------------Read the Running Time---------------------------
        file_name = file_name_base_results + "/RunningTimes/T_%s.txt" %file_name_ending
        T_run = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
        run_time[ensemble_count,itr] = sum(T_run)/len(T_run)
        #--------------------------------------------------------------------------
        
    itr = itr + 1
#==============================================================================


#===================CALCULATE THE MEAN AND VAR OF RESULTS======================

#-----------------------------Calculating the Mean-----------------------------
run_time_tot = run_time.mean(axis=0)
P_C_plus = det_P_C_plus.mean(axis=0)
P_C_minus = det_P_C_minus.mean(axis=0)
P_C_zero = det_P_C_zero.mean(axis=0)
Rec_plus = det_Rec_plus.mean(axis=0)
Rec_minus = det_Rec_minus.mean(axis=0)
Rec_zero = det_Rec_zero.mean(axis=0)

P_C_plus_hebb = det_P_C_plus_hebb.mean(axis=0)
P_C_minus_hebb = det_P_C_minus_hebb.mean(axis=0)
P_C_zero_hebb = det_P_C_zero_hebb.mean(axis=0)
Rec_plus_hebb = det_Rec_plus_hebb.mean(axis=0)
Rec_minus_hebb = det_Rec_minus_hebb.mean(axis=0)
Rec_zero_hebb = det_Rec_zero_hebb.mean(axis=0)


Prec_total_hebb = p_plus*P_C_plus_hebb+p_minus*P_C_minus_hebb+(1-connection_prob)*P_C_zero_hebb
Rec_total_hebb = p_plus * Rec_plus_hebb+p_minus*Rec_minus_hebb+(1-connection_prob)*Rec_zero_hebb

Prec_total = p_plus*P_C_plus+p_minus*P_C_minus+(1-connection_prob)*P_C_zero
Rec_total = p_plus * Rec_plus+p_minus*Rec_minus+(1-connection_prob)*Rec_zero
#------------------------------------------------------------------------------


#---------------------------Calculating the Variance---------------------------
var_P_C_plus = det_P_C_plus.var(axis=0)
var_P_C_minus = det_P_C_minus.var(axis=0)
var_P_C_zero = det_P_C_zero.var(axis=0)
var_Prec_total = Prec_total.var(axis=0)

var_Rec_plus = Rec_plus.var(axis=0)
var_Rec_minus = Rec_minus.var(axis=0)
var_Rec_zero = Rec_zero.var(axis=0)
var_Rec_total = Rec_total.var(axis=0)

var_P_C_plus_hebb = P_C_plus_hebb.var(axis=0)
var_P_C_minus_hebb = P_C_minus_hebb.var(axis=0)
var_P_C_zero_hebb = P_C_zero_hebb.var(axis=0)
var_Prec_total_hebb = Prec_total_hebb.var(axis=0)

var_Rec_plus_hebb = Rec_plus_hebb.var(axis=0)
var_Rec_minus_hebb = Rec_minus_hebb.var(axis=0)
var_Rec_zero_hebb = Rec_zero_hebb.var(axis=0)
var_Rec_total_hebb = Rec_total_hebb.var(axis=0)

var_run_time = run_time.var(axis=0)
#------------------------------------------------------------------------------
#==============================================================================



#==============================PLOT THE RESULTS================================
plot(range_T,P_C_plus,'b')
plot(range_T,P_C_minus,'r')
plot(range_T,P_C_zero,'g')

plot(range_T,Rec_plus,'b--')
plot(range_T,Rec_minus,'r--')
plot(range_T,Rec_zero,'g--')

#plot(range_T,P_C_plus_thr,'b--')
#plot(range_T,P_C_minus_thr,'r--')
#plot(range_T,P_C_zero_thr,'g--')
#==============================================================================


#=========================WRITE THE RESULTS TO THE FILE=========================

#--------------------------Construct Prpoper File Names-------------------------
file_name_ending = name_base + "Effect_q_n_exc_%s" %str(int(n_exc))
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
file_name = file_name_base_plot + "/Prec_exc_vs_T_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,P_C_plus[itr],var_P_C_plus[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Prec_inh_vs_T_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,P_C_minus[itr],var_P_C_minus[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Prec_zero_vs_T_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,P_C_zero[itr],var_P_C_zero[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()


file_name = file_name_base_plot + "/Prec_total_vs_T_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Prec_total[itr],var_Prec_total[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()



file_name = file_name_base_plot + "/Reca_exc_vs_T_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Rec_plus[itr],var_Rec_plus[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_inh_vs_T_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Rec_minus[itr],var_Rec_minus[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_zero_vs_T_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Rec_zero[itr],var_Rec_zero[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()


file_name = file_name_base_plot + "/Reca_total_vs_T_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Rec_total[itr],var_Rec_total[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()


file_name = file_name_base_plot + "/Reca_vs_Prec_exc_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t %f \t" %(Rec_plus[itr], P_C_plus[itr],var_Rec_plus[itr], var_P_C_plus[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_vs_Prec_inh_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t %f \t" %(Rec_minus[itr], P_C_minus[itr],var_Rec_minus[itr], var_P_C_minus[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_vs_Prec_zero_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t %f \t" %(Rec_zero[itr], P_C_zero[itr],var_Rec_zero[itr], var_P_C_zero[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_vs_Prec_total_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t %f \t" %(Rec_total[itr], Prec_total[itr],var_Rec_total[itr], var_Prec_total[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

plot(range_T,P_C_plus,'b')
plot(range_T,P_C_minus,'r')
plot(range_T,P_C_zero,'g')

plot(range_T,Rec_plus,'b--')
plot(range_T,Rec_minus,'r--')
plot(range_T,Rec_zero,'g--')






#-------------------------------------------------------------------------------



file_name = file_name_base_plot + "/Prec_exc_vs_T_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,P_C_plus_hebb[itr],var_P_C_plus_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Prec_inh_vs_T_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,P_C_minus_hebb[itr],var_P_C_minus_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Prec_zero_vs_T_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,P_C_zero_hebb[itr],var_P_C_zero_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()


file_name = file_name_base_plot + "/Prec_total_vs_T_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t" %(T,Prec_total_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()



file_name = file_name_base_plot + "/Reca_exc_vs_T_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Rec_plus_hebb[itr],var_Rec_plus_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_inh_vs_T_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Rec_minus_hebb[itr],var_Rec_minus_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_zero_vs_T_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Rec_zero_hebb[itr],var_Rec_zero_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()


file_name = file_name_base_plot + "/Reca_total_vs_T_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,Rec_total_hebb[itr],var_Rec_total_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()


file_name = file_name_base_plot + "/Reca_vs_Prec_exc_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t %f \t" %(Rec_plus_hebb[itr], P_C_plus_hebb[itr],var_Rec_plus_hebb[itr], var_P_C_plus_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_vs_Prec_inh_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t %f \t" %(Rec_minus_hebb[itr], P_C_minus_hebb[itr],var_Rec_minus_hebb[itr], var_P_C_minus_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_vs_Prec_zero_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t %f \t" %(Rec_zero_hebb[itr], P_C_zero_hebb[itr],var_Rec_zero_hebb[itr], var_P_C_zero_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

file_name = file_name_base_plot + "/Reca_hebb_vs_Prec_total_hebb_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t %f \t" %(Rec_total_hebb[itr], Prec_total_hebb[itr],var_Rec_total_hebb[itr], var_Prec_total_hebb[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()

#-------------------------------------------------------------------------------

file_name = file_name_base_plot + "/Running_time_%s.txt" %file_name_ending
running_time_file = open(file_name,'w')
itr = 0
for T in range_T:
    running_time_file.write("%f \t %f \t %f \t" %(T,run_time_tot[itr],var_run_time[itr]))
    running_time_file.write("\n")
    itr = itr + 1
running_time_file.close()



#==============================================================================
