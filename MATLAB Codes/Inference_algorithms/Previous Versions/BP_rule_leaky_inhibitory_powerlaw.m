function BP_rule_leaky_inhibitory_powerlaw(n_exc,n_inh,deg_min,deg_max,d_plus,d_minus,T,tau,theta,gamma,q,no_averaging_itrs,delta_plus,delta_minus,B_LLR_flag,mode,weight_rule,Delta,binary_mode)



%===========================INITIALIZATION=================================
n = n_exc + n_inh;
acc_ave_plus = 0;                    % average number of correctly infered excitatory edges
acc_ave_minus = 0;                    % average number of correctly infered inhibitory edges
acc_ave_0 = 0;                    % average number of correctly infered non-edges
err_ave = 0;                    % average number of mistakenly infered edges + missed edges
        
no_nodes_per_graph = 1;

% avg_no_edge = round(p*n);       % estimated number of edges in the graph
no_edges_fire = ceil(theta*n);  % estimated number of edges required for a neuron to fire

success_measure = zeros(1,3);
acc_theory = 0;
err_theory = 0;
% fit_coeff_0 = [-0.0000    0.0002   -0.0041    0.0362   -0.1104];
% fit_coeff_1 = [0.0001   -0.0073    0.2498   -3.8715   24.0011];

a=clock;                                % Initialize the seed for random number generation with the clock value.
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*a))); 
tempp = [];
%==========================================================================


%=======================AVERAGE OVER AN ENSEMBLE===========================
for av_itr = 1:no_averaging_itrs        
    
    t_last = 0;                     % The last time the output neuron fired
        
    %---------------------------Generate Activity--------------------------
    if ( (mode == 1) || (mode == 3) )         
        % params = [theta,q,tau,deg_min,deg_max,gamma];        
        params = [theta,q,tau,d_plus,d_minus,gamma];
    else
        params = [1];
    end
    
    %----------------------------------------------------------------------
    
    %----------------Generate the Graph One Node at a Time-----------------
    for node_itr = 1:no_nodes_per_graph
        [G,S,R] = generate_activity(1,n_exc,n_inh,[],T,mode,params,2);
        if (sum(abs(G)) == 0)
            while (sum(abs(G)) == 0)
                [G,S,R] = generate_activity(1,n_exc,n_inh,[],T,mode,params,2);
            end
        end
        %------------------Infer the Connectivity Matrix-------------------
        if (weight_rule == 1)
            W = zeros(1,n);
        else
            W = 0.5*ones(1,n);
        end
        
        
        for i = 1:n       
            t_last = 0;
            for jj = 1:T                            
                temp = 0;            
                temp_L = 0;
                temp_var = 0;
                temp_var_L = 0;
                for j = jj:-1:t_last+1                    
                    temp = temp + S(j,i)/tau^(1*(jj-j));
                    L = sum(S(j,:));                
                    temp_L = temp_L + L/tau^(1*(jj-j));
                    temp_var = temp_var + S(j,i)/tau^(2*(jj-j));
                    temp_var_L = temp_var_L + L/tau^(2*(jj-j));
                end
                
                  
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
            
            
                if (R(jj))                    
                    t_last = jj;                            
                end
            end        
        end    
        W2 = W;        
        %------------------------------------------------------------------

        %------------------Determine the Weight Threshold------------------
        % w_thr = determine_weight_thr_leaky(n,p,q,theta);
        % w_thr = q*(1/avg_T_fire+1/avg_T_fire2)/2;
        % W = (W>w_thr);        
        [W_sorted,ind] = sort(W);
        W = zeros(1,n);
        WW = zeros(1,n);     
        %      W(ind(end-sum(G>0)+1:end)) = abs(sign(W_sorted(end-sum(G>0)+1:end)));        
        %      W(ind(1:sum(G<0))) = -abs(sign(W_sorted(1:sum(G<0))));            
    
        
        aa = sum(sum(S))/T/n;        
        r = sum(R)/T;
        if (binary_mode <= 3)
            % [W,epsilon_plus_opt,epsilon_minus_opt] = epsilon_finder_powerlaw(W2,T,n_exc,n_inh,q,theta,tau,delta_plus,delta_minus,Delta,deg_min,deg_max,aa,r,binary_mode,weight_rule);
            [W,epsilon_plus_opt,epsilon_minus_opt,d_plus_min,d_minus_min] = epsilon_finder_powerlaw(W2,T,n_exc,n_inh,q,theta,tau,delta_plus,delta_minus,Delta,deg_min,deg_max,aa,r,binary_mode,weight_rule);
        else
        
            W = zeros(1,n);        
            W(ind(end-round(p_plus*n)+1:end)) = 1;
            W(ind(1:round(p_minus*n))) = -1;
        end
        %------------------------------------------------------------------
                
        %--------------------Calculate the Accuracy------------------------
        if (sum(G>0) > 0)
            acc_plus = sum((G>0).*(W>0))/sum(G>0);
            if (acc_plus > 1)
                error('What the h***?!')
            end
        else
            acc_plus = 1;
        end
        
        if (sum(G<0))
            acc_minus = sum((G<0).*(W<0))/sum(G<0);
            if (acc_minus > 1)
                error('What the h***?!')
            end
        else
            acc_minus = 1;
        end
        
        if (sum(G==0))
            acc_0 = sum((G==0).*(W==0))/sum(G==0);
            if (acc_0 > 1)
                error('What the h***?!')
            end
        else
            acc_0 = 1;
        end
        % err = abs(sum(sign(abs(W)+abs(G))-((G==W).*(abs(G)>0))));
        err = sum(W~=G);
        acc_ave_plus = acc_ave_plus + acc_plus;
        acc_ave_minus = acc_ave_minus + acc_minus;
        acc_ave_0 = acc_ave_0 + acc_0;
        
        err_ave = err_ave + err/(n-sum(abs(G)));        
        tempp = [tempp;acc_plus,acc_minus,acc_0];
        %------------------------------------------------------------------
    
    end
    111;
    
end
%==========================================================================


%===========================SAVE THE RESULTS===============================
success_measure(1) = acc_ave_plus/av_itr/node_itr;
success_measure(2) = acc_ave_minus/av_itr/node_itr;
success_measure(3) = acc_ave_0/av_itr/node_itr;   

% [p1_T,p2_T] = simple_hebb_rule_theory_leaky(n,p,q,T,theta,tau,avg_T_fire);
% simple_hebb_rule_theory(n,p,q,T,theta,0);    
% acc_theory(itr) =p1_T;    
% err_theory(itr) = p2_T;

if (B_LLR_flag)
    fid = fopen(['Simulation_Results/Power_law_Belief_LLR_leaky_inhib_n_',num2str(n_exc),'_',num2str(n_inh),'_no_averaging_itrs_',num2str(no_averaging_itrs),'_w_rule_',num2str(weight_rule),'_binary_',num2str(binary_mode),'_deg_',num2str(deg_min),'_',num2str(deg_max),'.txt'], 'a+');        
else
    fid = fopen(['Simulation_Results/Power_law_Belief_LLR_leaky_inhib_Delta_1_n_',num2str(n_exc),'_',num2str(n_inh),'_no_averaging_itrs_',num2str(no_averaging_itrs),'_w_rule_',num2str(weight_rule),'_binary_',num2str(binary_mode),'_deg_',num2str(deg_min),'_',num2str(deg_max),'.txt'], 'a+');
end
fprintf(fid, 'T \t %d \t theta \t %f \t g \t %f  \t d \t %d \t %d \t q \t %f \t tau \t %f \t delta \t %f \t %f \t acc_plus \t %f \t acc_minus \t %f \t acc_0 \t %f \t',T,theta,gamma,d_plus,d_minus,q,tau,delta_plus,delta_minus,success_measure(1),success_measure(2),success_measure(3));
fprintf(fid,'\n');
fclose(fid);
    
%==========================================================================

