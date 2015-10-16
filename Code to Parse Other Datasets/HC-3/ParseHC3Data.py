import matplotlib.pyplot as plt
import numpy as np
import csv


#-----------------------------Initializations-------------------------------
clusters_file_base = './Data/ec013.198.clu.'
spikes_file_base = './Data/ec013.198.res.'

sampling_freq = 20000.0
no_shanks = 8

task_name = 'ec013.18'
session_name = 'ec013.198'


neural_activity_file_write = './Data/HC3_ec013_198_processed.txt'

sp_times_tot_mat = []
spikes_tot = []
n_clusters = []
#----------------------------------------------------------------------------


#---------------------------Read the Spikes Matrix--------------------------
T_max = 0
for shank_no in range(1,no_shanks+1):
    spikes_file = spikes_file_base + str(shank_no)
    
    sp_times = np.genfromtxt(spikes_file, dtype=None)
    
    T = int(1000*max(sp_times)/sampling_freq) + 1
    
    if T > T_max:
        T_max = T

for shank_no in range(1,no_shanks+1):
    clusters_file = clusters_file_base + str(shank_no)
    spikes_file = spikes_file_base + str(shank_no)

    clusters_inds = np.genfromtxt(clusters_file, dtype=None)
    sp_times = np.genfromtxt(spikes_file, dtype=None)

    n = clusters_inds[0]-2                      # Clustes 0 and 1 do not correspond to any neuron
    
    sp_times_mat = np.zeros([n,T_max])
    sps = np.zeros([T_max,2])
    
    #pdb.set_trace()
    for i in range(0,len(sp_times)):        
        ii = clusters_inds[i+1]
        if ii > 1:
            tt = int(1000*sp_times[i]/sampling_freq)
            sp_times_mat[ii-2,tt] = 1
            #sps[tt,1] = tt/1000.0
            #sps[tt,0] = sum(n_clusters) + ii-2

    n_clusters.append(n)
    
    if shank_no>1:
        sp_times_tot_mat = np.vstack([sp_times_tot_mat,sp_times_mat])
        #spikes_tot = np.vstack([spikes_tot,sps])
    else:
        sp_times_tot_mat = sp_times_mat
        #spikes_tot = sps
#----------------------------------------------------------------------------

#--------------------------Verify the Imported Files-------------------------

#.....................Map Ours and Theirs Neuron IDs.........................
cell_id_inf = np.genfromtxt('./Data/hc3-cell.csv', dtype=str, delimiter=',')
cell_id_inf = cell_id_inf[cell_id_inf[:,1]==task_name]

neuron_id_map = {}

for item in cell_id_inf:
    cell_id = item[0]
    electrode = int(item[3])
    cluster = int(item[4]) - 2
    
    neuron_id = sum(n_clusters[0:electrode-1]) + cluster
    
    neuron_id_map[cell_id] = neuron_id
#............................................................................

#............................Retrieve Session ID.............................
sessions_inf = np.genfromtxt('./Data/hc3-session.csv', dtype=str, delimiter=',')
sessions_inf = sessions_inf[sessions_inf[:,2]==session_name]
session_id = int(sessions_inf[0,0])
#............................................................................

#.......................Read the Actual Spike Counts.........................
sp_counts = np.genfromtxt('./Data/hc3-spike_count.csv', dtype=int, delimiter=',')
sp_counts = sp_counts[sp_counts[:,1]==session_id]
#............................................................................


#...........................Verify the Results...............................
aa = np.sum(sp_times_tot_mat,axis = 1)
troublesome_entries = []
troublesome_values = []
for item in sp_counts:
    
    ind = neuron_id_map[str(item[0])]
    ff = int(item[2])
    if (abs(aa[ind] - ff)>1):
        troublesome_entries.append(ind)
        troublesome_values.append(abs(aa[ind] - ff))
        
if sum(troublesome_values)/float(sum(aa[troublesome_entries]))< 0.001:
    verify_flag = 1
else:
    verify_flag = 0
#............................................................................

#----------------------------------------------------------------------------


#------------------------Write the Spikes to the File------------------------
if verify_flag:
    spikes_tot = []
    inds = np.nonzero(sp_times_tot_mat)
    for i in range(297300,len(inds[0])):
        if i == 0:
            spikes_tot = [inds[0][i],inds[1][i]/1000.0]
        else:
            spikes_tot = np.vstack([spikes_tot,[inds[0][i],inds[1][i]/1000.0]])
            
    
    np.savetxt(neural_activity_file_write,spikes_tot,'%3.5f',delimiter='\t')    
#----------------------------------------------------------------------------

#-----------------Parse the Properties of Ground Truth Graph-----------------
# Each row in the following matrix described properties of one neuron
# The columns respectively represent: 1) ne: number of cells this cell monosynaptically excited
#                                     2) ni: number of cells this cell monosynaptically inhibited
#                                     3) eg: physiologically identified exciting cells based on CCG analysis
#                                     4) ig: physiologically identified inhibited cells based on CCG analysis
#                                     5) ed: based on cross-correlogram analysis, the cell is monosynaptically excited by other cells
#                                     6) id: based on cross-correlogram analysis, the cell is monosynaptically inhibited by other cells
#                                     7) re: Brain region (1 for EC3, 2 for EC4, 3 for EC5 and so on)
region_ind_map = {'EC3':1,'EC4':2,'EC5':3,'CA1':4,'CA3':5,'DG':6,'Unknown':7}

n = sum(n_clusters)
W_deg = np.zeros([n,7])
itr = 0
for item in cell_id_inf:
    W_deg[itr,0:6] = (item[6:12]).astype(int)
    W_deg[itr,6] = region_ind_map[item[5]]
    itr = itr + 1 

W_deg[:,5] = -W_deg[:,5]             # For the inhibitory neurons

#----------------------------------------------------------------------------