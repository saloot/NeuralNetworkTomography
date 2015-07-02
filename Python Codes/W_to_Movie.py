#=======================IMPORT THE NECESSARY LIBRARIES=========================
import numpy as np
import sys,getopt,os
import matplotlib.pyplot as plt
from PIL import Image
import scipy
#==============================================================================

#================================INITIALIZATIONS===============================

#--------------------------Initialize Other Variables--------------------------
n_exc = 160                             # Number of excitatory neurons in the network
n_inh = 40                              # Number of inhibitory neurons in the network
n = n_exc+ n_inh
connection_prob = 0.15                  # The probability that two neurons are connected
no_cascades = 8000                     # Number of stimulations performed on the network
frac_input_neurons = 0.4                # Fraction of initially stimulated neurons
delay_max = 1.0                         # This is done to avoid any incompatibilities in reading the data files


input_stimulus_freq = 20000             # The frequency of spikes by the neurons in the input layer (in Hz)

no_stimul_rounds = 1000
T_range = range(200, no_stimul_rounds, 150)
ensemble_count = 0                                     # The graph we use to make the movie from

height = 40                             # Number of rows in the submatrix of the connectivity matrix that we consider
width = 40                              # Number of columns in the submatrix of the connectivity matrix that we consider
W_trunc = np.zeros([height,width])      # The consdiered submatrix

file_name_base_results = "./Results"        # The folder to store resutls
adj_fact_exc = 0.75
adj_fact_inh = 0.5
#------------------------------------------------------------------------------

def image_whiten(imag,height,wdith):
    s = min(height,width)
    x = np.linspace(-s/2,s/2-1,width)
    y = np.linspace(-s/2,s/2-1,height)
    fx,fy=np.meshgrid(x,y)
    rho=np.sqrt(np.multiply(fx,fx)+np.multiply(fy,fy))
    f_0=0.4*s
    filt=np.multiply(rho,exp(-pow(rho/f_0,.4)))

    Iff=fft2(imag)
    imagew=real(ifft2(np.multiply(Iff,np.fft.fftshift(filt))))
    imagew2 = reshape(imagew,pow(s,2),1)

    imagew=np.sqrt(0.1)*imagew/imagew2.var()

    imagew3 = imagew - imagew.min();
    imagew3 = imagew3/imagew3.max();
    
    return imagew3
#==============================================================================


#=========================READ THE ORIGINAL GRAPH==============================

#-----------------------Construct Prpoper File Names---------------------------
Network = NeuralNet(no_layers,n_exc_array,n_inh_array,connection_prob_matrix,delay_max_matrix,random_delay_flag,'')
Network.read_weights(0,file_name_base_data)
file_name_ending23 = Network.file_name_ending + '_I_' + str(inference_method)
file_name_ending23 = file_name_ending23 + '_Loc_' + we_know_location
file_name_ending23 = file_name_ending23 + '_Pre_' + pre_synaptic_method
file_name_ending23 = file_name_ending23 + '_G_' + generate_data_mode
file_name_ending23 = file_name_ending23 + '_X_' + str(infer_itr_max)
if (sparsity_flag):
    file_name_ending23 = file_name_ending23 + '_S_' + str(sparsity_flag)

#------------------------------------------------------------------------------

#............Construct the Concatenated Weight Matrix..............            
W_tot = []
for l_in in range(0,Network.no_layers):
    n_exc = Network.n_exc_array[l_in]
    n_inh = Network.n_inh_array[l_in]
    n = n_exc + n_inh

    W_temp = []
    for l_out in range(0,Network.no_layers):
        n_exc = Network.n_exc_array[l_out]
        n_inh = Network.n_inh_array[l_out]
        m = n_exc + n_inh
                        
        if (l_out < l_in):
            if len(W_temp):
                W_temp = np.hstack([W_temp,np.zeros([n,m])])
            else:
                W_temp = np.zeros([n,m])
        else:
            ind = str(l_in) + str(l_out);
            temp_list = Network.Neural_Connections[ind];
            W = temp_list[0]
                        
            if len(W_temp):
                W_temp = np.hstack([W_temp,W])
            else:
                W_temp = W
                
                
                
    if len(W_tot):
        W_tot = np.vstack([W_tot,W_temp])
    else:
        W_tot = W_temp
#..................................................................
W_orig = W_tot

#============GENERATE RANDOM INDICES TO CONSIDER THEIR EVOLUTION===============
n,m = W_orig.shape
ind_horiz = np.random.randint(m, size=width)
ind_vert = np.random.randint(n, size=height)
#==============================================================================

    

W_orig_Trunk = W_orig[ind_vert,:]
W_orig_Trunk = W_orig_Trunk[:,ind_horiz]
W_orig_Trunk_W = image_whiten(W_orig_Trunk,height,width)
W_orig_Trunk = np.sign(W_orig_Trunk)
#W_orig_Trunk = W_orig_Trunk + 1
#W_orig_Trunk = W_orig_Trunk/2.0
#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
for T in T_range:
    
    #--------------------------------------------------------------------------                
    #--------------------READ THE NETWORK AND SPIKE TIMINGS--------------------

    #.......................Construct Prpoper File Names.......................    
    file_name_ending2 = file_name_ending23 + "_T_%s" %str(T)
    file_name_ending2 = file_name_ending2 + "_%s" %str(adj_fact_exc)
    file_name_ending2 = file_name_ending2 +"_%s" %str(adj_fact_inh)
    file_name_ending2 = file_name_ending2 + "_B_%s" %str(binary_mode)
    #..........................................................................
    
    #.......................Read the Graph from File...........................    
    file_name = file_name_base_results + "/Inferred_Graphs/W_Binary_%s.txt" %file_name_ending2
        
    W = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    #..........................................................................
    
    #............................Crop the Matrix...............................
    W_trunk = W[ind_vert,:]
    W_trunk = W_trunk[:,ind_horiz]
    #W_trunk_W = image_whiten(W_trunk,height,width)
    W_mean = np.zeros([1,width])
    W_mean[0,:] = W_trunk.mean(axis=0)
    #W_mean_orig = (n_exc-n_inh) * np.ones([1,height]) / float(n)
    #W_trunk = W_trunk - np.dot(np.ones([height,1]),W_mean) + np.dot(np.ones([height,1]),W_mean_orig)
    #W_trunk = W_trunk-W_trunk.min()
    #W_trunk = W_trunk - W_trunk.min()
    #W_trunk = W_trunk/float(W_trunk.max())
    #print W_trunk.mean(axis=0)
    #..........................................................................
    W_trunk = np.sign(W_trunk)
    file_name = './Results/Visualize_W_Movie/W_%s' %file_name_ending2
    file_name = file_name + '.png'
    scipy.misc.imsave(file_name, np.hstack([W_trunk,W_orig_Trunk]))
    
    #--------------Store the Error Between Actual and the Binary Image----------
    itr_mat = 0    
    save_matrix = np.zeros([30*40,3])
    for ik in range(0,30):
        for ij in range(0,40):
            val = W_trunk[ik,ij]-W_orig_Trunk[ik,ij]
            save_matrix[itr_mat,:] = [ij,ik,val]
            itr_mat = itr_mat + 1
    
    file_name = './Results/Visualize_W_Movie/W_error_%s.txt' %file_name_ending2
    np.savetxt(file_name,save_matrix,'%2.5f',delimiter='\t')
    #-----------------------------------------------------
    
    
#--------------------Save an image-----------------
itr_mat = 0
T = 950
save_matrix = np.zeros([30*40,3])
for ik in range(0,30):
    for ij in range(0,40):
        val = W_orig_Trunk[ik,ij]
        save_matrix[itr_mat,:] = [ij,ik,val]
        itr_mat = itr_mat + 1

file_name = './Results/Visualize_W_Movie/W_orig_%s.txt' %file_name_ending2
np.savetxt(file_name,save_matrix,'%2.5f',delimiter='\t')
#-----------------------------------------------------

#--------------------Save an image-----------------
itr_mat = 0
T = 950
save_matrix = np.zeros([30*40,3])
for ik in range(0,30):
    for ij in range(0,40):
        val = W_trunk[ik,ij]
        save_matrix[itr_mat,:] = [ij,ik,val]
        itr_mat = itr_mat + 1

file_name = './Results/Visualize_W_Movie/W_binary_%s.txt' %file_name_ending2
np.savetxt(file_name,save_matrix,'%2.5f',delimiter='\t')
#-----------------------------------------------------