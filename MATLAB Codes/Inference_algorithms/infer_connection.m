%==========================================================================
%=========================FUNCTION DESCRIPTION=============================
%==========================================================================
% This function infers the connections in a "real artificial" neural
% network from the recorded states. It returns the infered "real-valued"
% connections, which should be further processed to get the binary
% connectivity matrix
%==========================================================================
%==========================================================================


function W = infer_connection(n_exc,n_inh,S,R,T,weight_rule,tau,B_LLR_flag,Delta,synaptic_delay)

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


%=========================INFER THE CONNECTIONS============================    
for i = 1:network_size             
    t_last = 0;                                
    
    for jj = 1+synaptic_delay:T                                
        
        %---------------------Compute the Statistics-----------------------
        temp = 0;                        
        temp_L = 0;            
        temp_var = 0;            
        temp_var_L = 0;
            
        u_limit = max(1,jj-synaptic_delay);        
        l_limit = max(1,t_last+1-synaptic_delay);
            
        for j = u_limit:-1:l_limit        
            temp = temp + S(j,i)*exp(-(jj-synaptic_delay-j)/tau);            
            L = sum(S(j,:));                            
            temp_L = temp_L + L * exp(-(jj-synaptic_delay-j)/tau);            
            temp_var = temp_var + S(j,i) * exp(-2*(jj-synaptic_delay-j)/tau);            
            temp_var_L = temp_var_L + L * exp(-2 * (jj-synaptic_delay-j)/tau);            
        end
        %------------------------------------------------------------------
                   
        
        %-----------------------Update the Weights-------------------------                    
        if ( R(jj) && temp)
            if (B_LLR_flag)
                W(i) = W(i) + temp/temp_L;
            else
                W(i) = W(i) + Delta;                    
            end            
        elseif (~R(jj) && temp)
            if (B_LLR_flag)
                W(i) = W(i) - abs(temp/temp_L);
            else
                W(i) = W(i) - Delta;                    
            end            
        end
        %------------------------------------------------------------------    
            
        %-------------------------Update t_last----------------------------    
        if (R(jj))
            t_last = jj;                                        
        end
        %------------------------------------------------------------------    
        
    end
       
end
%==========================================================================    
   