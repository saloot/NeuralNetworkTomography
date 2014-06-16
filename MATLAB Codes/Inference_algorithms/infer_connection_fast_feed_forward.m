%==========================================================================
%=========================FUNCTION DESCRIPTION=============================
%==========================================================================
% This function infers the connections in a "real artificial" neural
% network from the recorded states. It returns the infered "real-valued"
% connections, which should be further processed to get the binary
% connectivity matrix
%==========================================================================
%==========================================================================


function W = infer_connection_fast_feed_forward(n_l1,n_l2,S_times,R,R_times,T,weight_rule,tau,B_LLR_flag,Delta,synaptic_delay)

%===============================INITIALIZATIONS============================
if (weight_rule == 1)
    W = zeros(1,n_l1);
else   
    W = 0.5*ones(1,n_l1);
end


if (weight_rule > 1)
    error('Sorry not tere yet!')
end
%==========================================================================


%==================PREPROCESS SPIKE TIMES AND MEMBRANE VOLTAGE=============
membrane_voltage_part = zeros(n_l1,T);
for i = 1:n_l1                 
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
for i = 1:n_l1
    
    %-------------------------Update the Weights---------------------------    
    if (B_LLR_flag)
        s = sum(membrane_voltage_part);
        % W(i) = - sum(((-1).^R).*((membrane_voltage_part(i,:)>.995)./(s+0.00001)));       
        temp =[0,diff(membrane_voltage_part(i,:))];
        % W(i) = sum((R>0).*((temp>.5)).*membrane_voltage_part(i,:));
        W(i) = sum((R(1:100)>0).*((membrane_voltage_part(i,1:100))));
    else
        W(i) = Delta*(-sum(((-1).^R).*membrane_voltage_part(i,:)>0)+sum(((-1).^R).*membrane_voltage_part(i,:)<0)/10);
        % sum(((-1).^R).*membrane_voltage_part>0)
        % sum(((-1).^R).*membrane_voltage_part<0)
    end
    %----------------------------------------------------------------------                       
end
%==========================================================================    
   