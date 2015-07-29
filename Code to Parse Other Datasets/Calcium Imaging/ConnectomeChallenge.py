import brainparse as bp
import matplotlib.pyplot as plt
import numpy as np

neural_activity_file = './data/fluorescence_mocktest.txt'
neural_activity_file_write = './data/fluorescence_mocktest_adapted.txt'

neural_connectivity_file = './data/network_mocktest.txt'
neural_connectivity_file_write = './data/network_mocktest_adapted.txt'
T = 50000
firing_thr = 0.12


#neural_activity_file = '../Data/fluorescence_mocktest.txt'
#---------------------------Read the Spikes Matrix--------------------------
series_dict = bp.parse_time_series(neural_activity_file)

spikes_tot = []
n_tot = len(series_dict)
spikes_tot_mat = np.zeros([n_tot,T])

for item in series_dict:
    sps = np.array(series_dict[item])
    temp = np.divide(np.array(range(0,20*len(sps),20)),1000)
    sps = sps + temp
    sps = np.reshape(sps,[1,len(sps)])
    if item > 0:
        spikes_tot = np.vstack([spikes_tot,sps])
    else:
        spikes_tot = sps
      
    #sps = sps[sps<T/1000.0]  
    #sps_inds = (1000*sps).astype(int)
    #spikes_tot_mat[item,sps_inds] = 1
#----------------------------------------------------------------------------


#------------------------Write the Spikes to the File------------------------
acc_file = open(neural_activity_file_write,'w')
sampling_int = 5
for item in series_dict:
    sps = np.array(series_dict[item])
    sps = (np.diff(sps) >= firing_thr).astype(int)
    sps_times[item,:] = sps
    #temp = np.divide(np.array(range(0,20*len(sps),20)),1000)
    #sps = sps + temp
    sps = np.multiply(sps,np.divide(np.array(range(1,sampling_int*len(sps),sampling_int)),1000.0))
    
    for i in range(0,len(sps)):
        if sps[i]>0:
            acc_file.write("%d \t %f\n" %(item,sps[i]))
    

acc_file.close()    
#----------------------------------------------------------------------------

#-------------------------Read the Ground Truth Graph------------------------
#neuron_connections, blocked = bp.parse_neuron_connections(neural_connectivity_file)
W_temp = np.genfromtxt(neural_connectivity_file, dtype=None, delimiter=',')
W = np.zeros([n_tot,n_tot])

for i in range(0,len(W_temp)):
    n_1 = W_temp[i,0]-1
    n_2 = W_temp[i,1]-1
    e = W_temp[i,2]
    W[n_1,n_2] = e

np.savetxt(neural_connectivity_file_write,W,'%1.5f',delimiter='\t')
#----------------------------------------------------------------------------        