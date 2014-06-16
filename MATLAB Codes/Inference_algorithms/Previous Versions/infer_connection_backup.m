%==========================================================================
%=========================FUNCTION DESCRIPTION=============================
%==========================================================================
% This function infers the connections in a "real artificial" neural
% network from the recorded states. It returns the infered "real-valued"
% connections, which should be further processed to get the binary
% connectivity matrix
%==========================================================================
%==========================================================================


function [W,W2]= infer_connection(n_exc,n_inh,S,R,T,weight_rule,p,tau,theta,B_LLR_flag,delta_plus,delta_minus,Delta,synaptic_delay)

%===============================INITIALIZATIONS============================
network_size = n_exc + n_inh;
no_edges_fire = ceil(theta*network_size);  % estimated number of edges required for a neuron to fire

if (weight_rule == 1)
    W = zeros(1,network_size);    
else   
    W = 0.5*ones(1,network_size);    
end

%--------------------Segregated Connection Probabilities-------------------
p_ave = p *(n_exc-n_inh)/network_size;
p_plus = p * n_exc/network_size;
p_minus = p * n_inh/network_size;
p_var = p_plus * (1-p_plus) + p_minus * (1-p_minus) + 2 *p_plus * p_minus;
%--------------------------------------------------------------------------

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
        
        %-----------------------Construct Beliefs--------------------------
        mu_plus = (1-p_ave)*temp + p_ave * temp_L;
        mu_zero = (-p_ave)*temp + p_ave * temp_L;
        mu_minus = (-p_ave-1)*temp + p_ave * temp_L;
            
        sigma_plus = sqrt(p_var*(temp_var_L - temp_var));
        sigma_zero = sigma_plus;            
        sigma_minus = sigma_plus;

            
        Q_plus = q_function(no_edges_fire,mu_plus,sigma_plus);
        Q_zero = q_function(no_edges_fire,mu_zero,sigma_zero);
        Q_minus = q_function(no_edges_fire,mu_minus,sigma_minus);
            
        Q_plus = R(jj)*Q_plus+(1-R(jj))*(1-Q_plus);
        Q_minus = R(jj)*Q_minus+(1-R(jj))*(1-Q_minus);
        Q_zero = R(jj)*Q_zero+(1-R(jj))*(1-Q_zero);
            
            
        Q = Q_plus + Q_zero + Q_minus;
        B_LLR1 = log(Q_plus/((Q-Q_plus)/2));
        B_LLR2 = log(Q_zero/((Q-Q_zero)/2));
        B_LLR3 = log(Q_minus/((Q-Q_minus)/2));
        B_LLR = [B_LLR1,B_LLR2,B_LLR3];
        %------------------------------------------------------------------
                
        
        %-----------------------Update the Weights-------------------------
        if (B_LLR_flag)
            if (weight_rule == 1)                    
                if ( (Q_plus > Q_zero) && (Q_plus > Q_minus) )                         
                    val = B_LLR1*temp;                                        
                elseif ( ( Q_minus > Q_plus) && ( Q_minus > Q_zero) )                         
                    val = -abs(B_LLR3)*temp;                    
                else                    
                    val = -abs(B_LLR2)*sign(W(i))/2;
                    val = 0;                    
                end                
            else                
                upd = -temp*((-1)^(R(jj)));                
                if (upd > 0)                                        
                    val = (1+Delta)^(B_LLR1);                    
                elseif (upd < 0)                        
                    val = 1/((1+Delta)^(B_LLR3));                    
                else                    
                    val = 1.00^(sign(0.5-W(i)));                    
                end                
            end            
        else            
            if (weight_rule == 1)            
                upd = -temp*((-1)^(R(jj)));                           
                if (upd > delta_plus)                            
                    val = Delta;                                                            
                elseif (upd < -delta_minus)                                            
                    val = -Delta;                                    
                else                    
                    val = -Delta*sign(W(i))/2;                    
                    val = 0;                    
                end                
            else                
                upd = -temp*((-1)^(R(jj)));
                                    
                if (upd > delta_plus)                
                    val = (1+Delta);                    
                elseif (upd < -delta_minus)                        
                    val = 1/(1+Delta);                    
                else                    
                    val = 1.00^(sign(0.5-W(i)));
                end                
            end            
        end
        
        
        
        
        if (weight_rule == 1)        
            W(i) = W(i) + val;                                                    
        else            
            W(i) = W(i) * val;            
        end                    
        %------------------------------------------------------------------    
            
        %-------------------------Update t_last----------------------------    
        if (R(jj+so))                                    
            t_last = jj;                                        
        end
        %------------------------------------------------------------------    
        
    end
       
end
W2 = W;
%==========================================================================    
   