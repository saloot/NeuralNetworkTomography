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

T_range = range(10, no_cascades, 200)                  # The range of sample sizes considered to investigate the effect of sample size on the performance
ensemble_count = 0                                     # The graph we use to make the movie from

height = 40                             # Number of rows in the submatrix of the connectivity matrix that we consider
width = 40                              # Number of columns in the submatrix of the connectivity matrix that we consider
W_trunc = np.zeros([height,width])      # The consdiered submatrix

file_name_base_results = "./Results/Recurrent/Inferred_Graphs/"        # The folder to store resutls
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

#============GENERATE RANDOM INDICES TO CONSIDER THEIR EVOLUTION===============
ind_horiz = np.random.randint(n_exc+n_inh, size=width)
ind_vert = np.random.randint(n_exc+n_inh, size=height)
#==============================================================================

#=========================READ THE ORIGINAL GRAPH==============================

#-----------------------Construct Prpoper File Names---------------------------
file_name_ending = "n_exc_%s" %str(int(n_exc))
file_name_ending = file_name_ending + "_n_inh_%s" %str(int(n_inh))    
file_name_ending = file_name_ending + "_p_%s" %str(connection_prob)
file_name_ending = file_name_ending + "_r_%s" %str(frac_input_neurons)
#file_name_ending = file_name_ending + "_f_%s" %str(input_stimulus_freq)
file_name_ending = file_name_ending + "_d_%s" %str(delay_max)
file_name_ending = file_name_ending + "_T_%s" %str(no_cascades)    
file_name_ending = file_name_ending + "_%s" %str(ensemble_count)
#------------------------------------------------------------------------------


file_name = "./Data/Recurrent/Graphs/We_Recurrent_Cascades_Delay_%s.txt" %file_name_ending
We = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
file_name = "./Data/Recurrent/Graphs/Wi_Recurrent_Cascades_Delay_%s.txt" %file_name_ending
Wi = np.genfromtxt(file_name, dtype=None, delimiter='\t')
    
W_orig = np.vstack((We,Wi))
W_orig_Trunk = W_orig[ind_vert,:]
W_orig_Trunk = W_orig_Trunk[:,ind_horiz]
W_orig_Trunk_W = image_whiten(W_orig_Trunk,height,width)
W_orig_Trunk = np.sign(W_orig_Trunk)
W_orig_Trunk = W_orig_Trunk + 1
W_orig_Trunk = W_orig_Trunk/2.0
#==============================================================================


#================PERFORM THE INFERENCE TASK FOR EACH ENSEMBLE==================
for T in T_range:
    
    #--------------------------------------------------------------------------                
    #--------------------READ THE NETWORK AND SPIKE TIMINGS--------------------

    #.......................Construct Prpoper File Names.......................    
    file_name_ending2 = file_name_ending + "_%s" %str(T)
    #..........................................................................
    
    #.......................Read the Graph from File...........................
    file_name = file_name_base_results + "W_Recurrent_Cascades_Delay_%s.txt" %file_name_ending2
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
    W_trunk = W_trunk-W_trunk.min()
    #W_trunk = W_trunk - W_trunk.min()
    #W_trunk = W_trunk/float(W_trunk.max())
    #print W_trunk.mean(axis=0)
    #..........................................................................
    
    file_name = './Results/Visualize_W_Movie/W_%s' %file_name_ending2
    file_name = file_name + '.png'
    scipy.misc.imsave(file_name, np.hstack([W_trunk,W_orig_Trunk]))
