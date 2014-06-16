%==========================================================================
%=========================FUNCTION DESCRIPTION=============================
%==========================================================================
% This function infers the connections in a "real artificial" neural
% network from the recorded states. It returns the infered "real-valued"
% connections, which should be further processed to get the binary
% connectivity matrix
%==========================================================================
%==========================================================================


function W = infer_connection_fast(n_exc,n_inh,S_times,R,R_times,T,weight_rule,tau,B_LLR_flag,Delta,synaptic_delay)

%===============================INITIALIZATIONS============================
network_size = n_exc + n_inh;

if (weight_rule == 1)
    W = zeros(1,network_size);    
else   
    W = 0.5*ones(1,network_size);    
end


if (weight_rule > 1)
    error('Sorry not tere yet!')
end
%==========================================================================


%==================PREPROCESS SPIKE TIMES AND MEMBRANE VOLTAGE=============
membrane_voltage_part = zeros(network_size,T);
for i = 1:network_size                 
    F = S_times{i};
    F = F + synaptic_delay;
    % F = min(F + synaptic_delay,T);
    
    t = [1:T];
    
    %---Constructu Cummulative Spike Timing for the Pre-Synaptic Neuron----
    
    for j = 1:length(F)
        temp = (t - F(j)>=0).*exp(-max(t - F(j),0)/tau);
        
        t_temp = (R_times-F(j)>0);
        ind = find(t_temp, 1, 'first');
        temp(R_times(ind)+1:end) = 0;
        membrane_voltage_part(i,:) = membrane_voltage_part(i,:) + temp;
    end
    %----------------------------------------------------------------------
end
%==========================================================================

%=========================INFER THE CONNECTIONS============================    
for i = 1:network_size                 
    
    %-------------------------Update the Weights---------------------------    
    if (B_LLR_flag)
        s = sum(membrane_voltage_part);
        % W(i) = - sum(((-1).^R).*((membrane_voltage_part(i,:)>.995)./(s+0.00001)));       
        temp =[0,diff(membrane_voltage_part(i,:))];
        W(i) = sum((R>0).*((temp>.95)).*membrane_voltage_part(i,:))./sum((R==0).*((temp>.95)).*membrane_voltage_part(i,:));
    else
        W(i) = Delta*(-sum(((-1).^R).*membrane_voltage_part(i,:)>0)+sum(((-1).^R).*membrane_voltage_part(i,:)<0)/10);
        % sum(((-1).^R).*membrane_voltage_part>0)
        % sum(((-1).^R).*membrane_voltage_part<0)
    end
    %----------------------------------------------------------------------                       
end
%==========================================================================    
   