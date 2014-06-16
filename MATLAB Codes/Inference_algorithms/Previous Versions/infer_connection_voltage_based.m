% function [W,W2]= infer_connection_voltage_based(n_exc,n_inh,S,T,weight_rule,p,theta,B_LLR_flag,delta_plus,delta_minus,Delta,V)
acc_ave_plus = 0;
acc_ave_minus = 0;
acc_ave_0 = 0;
thr_exc = 0.04;
thr_inh = -0.06;

for ensmeble_count = 1:100  
for neuron_no = 1:n
    V = read_votltages(n_exc,n_inh,ensmeble_count,T);
    S = read_spikes(n_exc,n_inh,ensmeble_count,T);
    G = read_graph(n_exc,n_inh,ensmeble_count);
    g = G(:,neuron_no)';
    
    n = n_exc + n_inh;
    no_edges_fire = ceil(theta*n);  % estimated number of edges required for a neuron to fire
    if (weight_rule == 1)
        W = zeros(1,n);
    else
        W = 0.5*ones(1,n);
    end

    no_identifiable_links = zeros(1,n);
    v = V(neuron_no,:);
    arrival_times = detect_arriving_spikes(v,thr_exc,thr_inh);
    for i = 1:n  
        
        s = S(i,:);
        if ( sum(s) )
            no_identifiable_links(i) = 1;
            if (g(i)>0)
                111;
            end
        end
        W_temp = zeros(1,14);
        for synaptic_delay = 0:13
            s_shifted = circshift(s,synaptic_delay);
            s_shifted(1:synaptic_delay) = 0;            
                                    
            if (weight_rule == 1)
                W_temp(synaptic_delay+1) = s_shifted*arrival_times';
            else                
                W_temp(synaptic_delay+1) = 0.5 * Delta^(s_shifted*arrival_times');
            end                    
        end        
        W(i) = sign(sum(W_temp)/(synaptic_delay+1));
    end    
    W2 = W;        
    acc_plus = sum((g>0).*(W>0))/sum(g>0);
    acc_minus = sum((g<0).*(W<0))/sum(g<0);
    acc_0 = sum((g==0).*(W==0))/sum(g==0);

    acc_plus_effective = sum((g>0).*(W>0))/(sum((g>0).*no_identifiable_links));
    acc_minus_effective = sum((g<0).*(W<0))/sum((g<0).*no_identifiable_links);
    
    acc_ave_plus = acc_ave_plus + acc_plus_effective;
    acc_ave_minus = acc_ave_minus + acc_minus_effective;
    acc_ave_0 = acc_ave_0 + acc_0;   
end
%--------------------------------------------------------------------------
end

acc_ave_plus = acc_ave_plus/neuron_no/ensmeble_count;
acc_ave_minus = acc_ave_minus/neuron_no/ensmeble_count;
acc_ave_0 = acc_ave_0/neuron_no/ensmeble_count;