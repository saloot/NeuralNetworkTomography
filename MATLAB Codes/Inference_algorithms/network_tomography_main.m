%==========================================================================
%============================CODE DESCRIPTION==============================
%==========================================================================
% This code infers the connections in a "real artificial" neural network 
% from the recorded states. It returns the infered binary connectivity
% matrix and calculates the accuracy of the proposed algorithm over a
% number of randomly generated instances.
%==========================================================================
%==========================================================================


%===============================INITIALIZATIONS============================
n_exc = 320;                        % Number of exctatory neurons in the network
n_inh = 80;                         % Number of ihibitory neurons in the network

n_f = 800;                          % The number of neuron in the first layer of a feed-forward network
n_inp = 800;                        % The number of neurons in the first layer of the feed-forward netwrok that will be stimulated
n_o = n_exc + n_inh;                % The number of neurons in the second layer of the feed-forward network
input_stim_freq = 800;              % The frequency of input poisson stimulus
FF_flag = 1;                        % If 0, the second layer will have recurrent connections as well. Otherwise, we will only have feed-forward connections

T = 10000;                          % Number of recorded samples
if FF_flag
    network_size = n_f;
else
    network_size = n_exc + n_inh;
end

ensemble_size = 10;                  % The number of radomly generated networks

p = 0.15;                           % The connection probability

if ~FF_flag
    p_exc = p * n_exc/network_size;     % The effective probability of having an excitatory connection
    p_inh = p * n_inh/network_size;     % The effective probability of having an inhibitory connection
else
    p_exc = p;
    p_inh = 0;
end

tau_syn = 20;                      % The time constant of the synapses in 0.1ms
synaptic_delay = 0;                % The time it takes for the neural pulse to go through a neuron i 0.1 ms
B_LLR_flag = 1;                     % Determines if the update in weights is constant (1) or proportional to the beliefs (2)
weight_rule = 1;                    % The way the weights will be updated, 1 for additive, 2 for multlipicative
Delta = 0.01;                       % The amount of update in the weights if B_LLR_flag = 1

acc_ave_plus = 0;                    % average number of correctly infered excitatory edges
acc_ave_minus = 0;                    % average number of correctly infered inhibitory edges
acc_ave_zero = 0;                    % average number of correctly infered non-edges
neuron_ind = 1;


    
% addpath(genpath('/home1/amir/cluster/Common_Library'))
addpath(genpath('../../Neural_Network_Tomography'))

data_file_path = '../Data';
%==========================================================================


for ensmeble_count = 0:ensemble_size-1
%=======================READ NEURAL ACTIVITY FROM FILE=====================
    
%------------------------Read the Connectivity Graph-----------------------
if FF_flag
    mode = 3;
    params{1} = [n_f,n_o,n_inp,p,synaptic_delay,input_stim_freq];
    params{2} = data_file_path;

end
G = read_graph(ensmeble_count,mode,params);
g = G(:,neuron_ind);
%--------------------------------------------------------------------------
                    
%------------------------Read the Recorded States--------------------------
if FF_flag
    mode = 3;
    params{1} = [n_f,n_o,n_inp,p,synaptic_delay,T,input_stim_freq];
    params{2} = data_file_path;
end

S = read_spikes(ensmeble_count,params,mode);
S_l1 = S{1};
S_l2 = S{2};

S_l1 = S_l1';
S_l2 = S_l2';
S_times = read_spikes_v2(ensmeble_count,params,mode);

S_times_l1 = S_times{1};
S_times_l2 = S_times{2};

R_times = S_times_l2{neuron_ind};
R = S_l2(:,neuron_ind)';
%--------------------------------------------------------------------------

%==========================================================================


%=========================INFER THE CONNECTIONS============================    
synaptic_delay = 10;
if FF_flag
    W = infer_connection_fast_feed_forward(n_f,n_o,S_times_l1,R,R_times,T,weight_rule,tau_syn,B_LLR_flag,Delta,synaptic_delay);
else
    W = infer_connection_fast(n_exc,n_inh,S_times,R,R_times,T,weight_rule,tau_syn,B_LLR_flag,Delta,synaptic_delay);
end
%==========================================================================


%=====================TRANSFORM THE CONNECTIONS TO BINARY==================
% [W,epsilon_plus_opt,epsilon_minus_opt,d_plus_min,d_minus_min] = real_to_binary_graph(W2,T,n_exc,n_inh,q,theta,tau,delta_plus,delta_minus,Delta,deg_min,deg_max,aa,r,binary_mode,weight_rule);
[W_sorted,ind] = sort(W);
W_binary = zeros(network_size,1);
W_binary(ind(end-round(p_exc * network_size)+1:end)) = 1;
W_binary(ind(1:round(p_inh * network_size))) = -1;
%==========================================================================


%=================CALCULATE THE ACCURRACY OF THE ALGORITHM=================
if (sum(g>0) > 0)            
    acc_plus = sum((g>0).*(W_binary>0))/sum(g>0);
else    
    acc_plus = 1;
end

if (sum(g<0))
    acc_minus = sum((g<0).*(W_binary<0))/sum(g<0);
else    
    acc_minus = 1;   
end

if (sum(g==0))
    acc_zero = sum((g==0).*(W_binary==0))/sum(g==0);
else    
    acc_zero = 1;
end

acc_ave_plus = acc_ave_plus + acc_plus;
acc_ave_minus = acc_ave_minus + acc_minus;       
acc_ave_zero = acc_ave_zero + acc_zero;
111
end
%==========================================================================

acc_ave_plus = acc_ave_plus/ensemble_size;
acc_ave_minus = acc_ave_minus/ensemble_size;
acc_ave_zero   = acc_ave_zero/ensemble_size;