from brian import *
from brian.library.IF import *
from brian.library.synapses import *
import time
import numpy as np 

#================================INITIALIZATIONS===============================
n_exc = 320
n_inh = 80
n = n_exc + n_inh
connection_prob = 0.15
fract_input_exc = 100;
fract_input_inh = 2;


C = 200 * pF
taum = 10 * msecond
gL = C / taum
EL = -70 * mV
VT = -55 * mV
DeltaT = 3 * mV

# Synapse parameters
Ee = 0 * mvolt
Ei = -80 * mvolt
taue = 5 * msecond
taui = 10 * msecond
no_samples = 20000
running_period = int(no_samples/10000)


eqs = exp_IF(C, gL, EL, VT, DeltaT)
# Two different ways of adding synaptic currents:
eqs += Current('''
Ie=ge*(Ee-vm) : amp
dge/dt=-ge/taue : siemens
''')
eqs += exp_conductance('gi', Ei, taui) # from library.synapses

ensemble_size = 1
#==============================================================================




for ensemble_count in range(0,ensemble_size):

#============================GENERATE THE NETWORK==============================
    P = NeuronGroup(n, model=eqs, threshold= -20 * mvolt, reset=EL, refractory=2 * ms)
    Pe = P.subgroup(n_exc)
    Pi = P.subgroup(n_inh)
    we = 1.5 * nS # excitatory synaptic weight
    wi = 2.5 * we # inhibitory synaptic weight
    Ce = Connection(Pe, P, 'ge', weight=we, sparseness=connection_prob)
    Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=connection_prob)
    
    #------------------Save Connectivity Matrices to the File----------------------
    We = Ce.W.todense()
    Wi = Ci.W.todense()
    
    #np.savetxt("/Hesam/Academic/Network Tomography/We_%s.txt" %str(ensemble_count),We,'%1.4f',delimiter='\t',newline='\n')
    #np.savetxt("/Hesam/Academic/Network Tomography/Wi_%s.txt" %str(ensemble_count),Wi,'%1.4f',delimiter='\t',newline='\n')
#------------------------------------------------------------------------------

    # Initialization
    P.vm = randn(len(P)) * 10 * mV - 70 * mV
    P.ge = (randn(len(P)) * 2 + 5) * we
    P.gi = (randn(len(P)) * 2 + 5) * wi

    # Excitatory input to a subset of excitatory and inhibitory neurons
    # Excitatory neurons are excited for the first 200 ms
    # Inhibitory neurons are excited for the first 100 ms
    input_layer1 = Pe.subgroup(fract_input_exc)
    input_layer2 = Pi.subgroup(fract_input_inh)
    input1 = PoissonGroup(fract_input_exc, rates=lambda t: (t < 200 * ms and 2000 * Hz) or 100 * Hz)
    input2 = PoissonGroup(fract_input_inh, rates=lambda t: (t < 10 * ms and 2000 * Hz) or 0 * Hz)
    input_co1 = IdentityConnection(input1, input_layer1, 'ge', weight=we)
    input_co2 = IdentityConnection(input2, input_layer2, 'ge', weight=we)

    # Record the number of spikes   
    M = SpikeMonitor(P)
    run(1500 * ms)

    M.clock.reinit()
    
    print M.nspikes / n, "spikes per neuron"
    
#------------------------Save Spike Times to the File--------------------------
    S_time_file = open("/Hesam/Academic/Network Tomography/S_times_%s.txt" %str(ensemble_count),'w')

    SS = M.spiketimes
    for item in SS.items():
        if len(item)>1:
            itt = item[1]
            for j in range(0,len(itt)-1):
                S_time_file.write("%s \t" % itt[j])
            S_time_file.write("-1 \n")
#------------------------------------------------------------------------------

    M.reinit()