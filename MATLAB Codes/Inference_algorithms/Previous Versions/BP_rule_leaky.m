function BP_rule_leaky(n,T,tau,theta,p,q,no_averaging_itrs,B_LLR_thr,B_LLR_flag)

%===========================INITIALIZATION=================================
acc_ave = 0;                    % average number of correctly infered edges
err_ave = 0;                    % average number of mistakenly infered edges + missed edges
        
avg_no_edge = round(p*n);       % estimated number of edges in the graph
no_edges_fire = ceil(theta*n);  % estimated number of edges required for a neuron to fire

success_measure = zeros(1,2);
acc_theory = 0;
err_theory = 0;
% fit_coeff_0 = [-0.0000    0.0002   -0.0041    0.0362   -0.1104];
% fit_coeff_1 = [0.0001   -0.0073    0.2498   -3.8715   24.0011];

a=clock;                                % Initialize the seed for random number generation with the clock value.
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*a))); 
%==========================================================================


%=======================AVERAGE OVER AN ENSEMBLE===========================
for av_itr = 1:no_averaging_itrs        
    
    t_last = 0;                     % The last time the output neuron fired
        
    %----------------------Create an Erdos Reny Random Graph---------------    
    G = erdos_reny(1,n,p);        
    %----------------------------------------------------------------------
                
    %------------------------Record Sample States--------------------------
    S = zeros(T,n);                 % recorded stimulus states         
    R = zeros(T,1);                 % recorder output state            
                
    for i = 1:T    
        s = max(rand(1,n) < q,0);   % stimulus applied to n neurons            
        S(i,:) = s';                                
        denom = tau.^(0:1:i-t_last-1);                        
        h = sum(G*(S(i:-1:t_last+1,:)')./denom);
            
        R(i) = 1*(h/n > theta);        
    end    
    %----------------------------------------------------------------------
                
    %--------------------Infer the Connectivity Matrix---------------------        
    W = zeros(1,n);
    
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
                
            mu_plus = (1-p)*temp + p * temp_L;
            mu_minus = (-p)*temp + p * temp_L;
            sigma = sqrt(p*(1-p)*(temp_var_L - temp_var));
            Q_plus = q_function(no_edges_fire,mu_plus,sigma);
            Q_minus = q_function(no_edges_fire,mu_minus,sigma);
            
            if (R(jj))                
                B_LLR = log(Q_plus/Q_minus);
                if (B_LLR>B_LLR_thr)                            
                    if (B_LLR_flag)                                
                        W(i) = W(i) + B_LLR * temp;                            
                    else                        
                        W(i) = W(i) + 1 * temp;
                    end                    
                end                
            else                
                B_LLR = log((1-Q_plus)/(1-Q_minus));
                if (S(jj,i))
                    111;
                end
                if (B_LLR>B_LLR_thr)                            
                    if (B_LLR_flag)                                
                        W(i) = W(i) + B_LLR * temp;                            
                    else                        
                        W(i) = W(i) + 1 * temp;
                    end                    
                end                
            end
            if (R(jj))                    
                t_last = jj;                            
            end
        end        
    end    
    W2 = W;        
    %----------------------------------------------------------------------

    %--------------------Determine the Weight Threshold--------------------        
    [W_sorted,ind] = sort(W);
     % w_thr = determine_weight_thr_leaky(n,p,q,theta);
     % w_thr = q*(1/avg_T_fire+1/avg_T_fire2)/2;
     % W = (W>w_thr);        
     
     W = zeros(1,n);     
     W(ind(end-sum(G)+1:end)) = abs(sign(W_sorted(end-sum(G)+1:end)));        
     %---------------------------------------------------------------------
                
     %----------------------Calculate the Accuracy-------------------------
     acc = sum(W.*G);        
     err = abs(sum(sign(W+G)-W.*G));        
     acc_ave = acc_ave + acc/sum(G);        
     err_ave = err_ave + err/(n-sum(G));        
     %---------------------------------------------------------------------
    
end
%==========================================================================


%===========================SAVE THE RESULTS===============================
success_measure(1) = acc_ave/av_itr;
success_measure(2) = err_ave/av_itr;   
% [p1_T,p2_T] = simple_hebb_rule_theory_leaky(n,p,q,T,theta,tau,avg_T_fire);
% simple_hebb_rule_theory(n,p,q,T,theta,0);    
% acc_theory(itr) =p1_T;    
% err_theory(itr) = p2_T;

if (B_LLR_flag)
    fid = fopen(['Simulation_Results/Belief_LLR_leaky_n_',num2str(n),'_no_averaging_itrs_',num2str(no_averaging_itrs),'.txt'], 'a+');        
else
    fid = fopen(['Simulation_Results/Belief_LLR_leaky_Delta_1_n_',num2str(n),'_no_averaging_itrs_',num2str(no_averaging_itrs),'.txt'], 'a+');        
end
fprintf(fid, 'T \t %d \t theta \t %f \t p \t %f \t q \t %f \t tau \t %f \t BBLR_thr \t %f \t acc \t %f \t err \t %f \t',T,theta,p,q,tau,B_LLR_thr,success_measure(1),success_measure(2));
fprintf(fid,'\n');
fclose(fid);
    
%==========================================================================

