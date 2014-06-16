

%===========================INITIALIZATION=================================
acc_ave_plus = 0;                    % average number of correctly infered excitatory edges
acc_ave_minus = 0;                    % average number of correctly infered inhibitory edges
acc_ave_0 = 0;                    % average number of correctly infered non-edges
err_ave = 0;                    % average number of mistakenly infered edges + missed edges
        
avg_no_edge = round(p*n);       % estimated number of edges in the graph
no_edges_fire = ceil(theta*n);  % estimated number of edges required for a neuron to fire

success_measure = zeros(1,3);

a=clock;                                % Initialize the seed for random number generation with the clock value.
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*a))); 
tempp = [];
%==========================================================================


%=======================AVERAGE OVER AN ENSEMBLE===========================
for av_itr = 2:no_averaging_itrs        
    
    t_last = 0;                     % The last time the output neuron fired
        
    %---------------------------Generate Activity--------------------------
    if ( (mode == 1) || (mode == 3) ) 
        params = [theta,q,tau];
    elseif (mode == 4)
        params = av_itr;
    else
        params = [1];
    end
    [G,S,R] = generate_activity(1,n_exc,n_inh,p,T,mode,params);
    %----------------------------------------------------------------------
                
    %--------------------Infer the Connectivity Matrix---------------------        
    if (mode == 4)
        for i = 1:n
            g = G(:,i);
            g = g';
            R = S(:,i);
            [W,W2] = infer_connection(n_exc,n_inh,S,R,T,weight_rule,p,tau,theta,B_LLR_flag,delta_plus,delta_minus,Delta,synaptic_delay);
            deg_max = round(2*p*n);
            aa = sum(sum(S))/T/n;
            r = sum(R)/T;
            [W,epsilon_plus_opt,epsilon_minus_opt] = epsilon_finder(W2,T,n_exc,n_inh,q,theta,delta_plus,delta_minus,Delta,deg_min,deg_max,aa,r,binary_mode,weight_rule);
            %----------------------Calculate the Accuracy-------------------------
            acc_plus = sum((g>0).*(W>0))/sum(g>0);
            acc_minus = sum((g<0).*(W<0))/sum(g<0);
            acc_0 = sum((g==0).*(W==0))/sum(g==0);
            
            acc_ave_plus = acc_ave_plus + acc_plus;
            acc_ave_minus = acc_ave_minus + acc_minus;
            acc_ave_0 = acc_ave_0 + acc_0;     
            %---------------------------------------------------------------------

        end
    else
        [W,W2]= infer_connection(n_exc,n_inh,S,R,T,weight_rule,p,tau,theta,B_LLR_flag,delta_plus,delta_minus,Delta,synaptic_delay);
        deg_max = round(2*p*n);
        aa = sum(sum(S))/T/n;
        r = sum(R)/T;
        [W,epsilon_plus_opt,epsilon_minus_opt] = epsilon_finder(W2,T,n_exc,n_inh,q,theta,delta_plus,delta_minus,Delta,deg_min,deg_max,aa,r,binary_mode,weight_rule);

        %----------------------Calculate the Accuracy-------------------------
        acc_plus = sum((G>0).*(W>0))/sum(G>0);
        acc_minus = sum((G<0).*(W<0))/sum(G<0);
        acc_0 = sum((G==0).*(W==0))/sum(G==0);
     
        acc_ave_plus = acc_ave_plus + acc_plus;
        acc_ave_minus = acc_ave_minus + acc_minus;
        acc_ave_0 = acc_ave_0 + acc_0;          
        %---------------------------------------------------------------------
    end
    
    
end
%==========================================================================


%===========================SAVE THE RESULTS===============================
if (mode == 4)
    success_measure(1) = acc_ave_plus/av_itr/n;
    success_measure(2) = acc_ave_minus/av_itr/n;
    success_measure(3) = acc_ave_0/av_itr/n;   
else
    success_measure(1) = acc_ave_plus/av_itr;
    success_measure(2) = acc_ave_minus/av_itr;
    success_measure(3) = acc_ave_0/av_itr;   
end

% [p1_T,p2_T] = simple_hebb_rule_theory_leaky(n,p,q,T,theta,tau,avg_T_fire);
% simple_hebb_rule_theory(n,p,q,T,theta,0);    
% acc_theory(itr) =p1_T;    
% err_theory(itr) = p2_T;

if (mode == 4)
    if (B_LLR_flag)
        fid = fopen(['Simulation_Results/Belief_LLR_leaky_inhib_n_',num2str(n_exc),'_',num2str(n_inh),'_no_averaging_itrs_',num2str(no_averaging_itrs),'_w_rule_',num2str(weight_rule),'_binary_',num2str(binary_mode),'_mode_4.txt'], 'a+');        
    else
        fid = fopen(['Simulation_Results/Belief_LLR_leaky_inhib_Delta_1_n_',num2str(n_exc),'_',num2str(n_inh),'_no_averaging_itrs_',num2str(no_averaging_itrs),'_w_rule_',num2str(weight_rule),'_binary_',num2str(binary_mode),'_mode_4.txt'], 'a+');
    end
    fprintf(fid, 'T \t %d \t delta \t %f \t %f \t acc_plus \t %f \t acc_minus \t %f \t acc_0 \t %f \t',T,delta_plus,delta_minus,success_measure(1),success_measure(2),success_measure(3));
    fprintf(fid,'\n');
    fclose(fid);
else
    if (B_LLR_flag)
        fid = fopen(['Simulation_Results/Belief_LLR_leaky_inhib_n_',num2str(n_exc),'_',num2str(n_inh),'_no_averaging_itrs_',num2str(no_averaging_itrs),'_w_rule_',num2str(weight_rule),'_binary_',num2str(binary_mode),'.txt'], 'a+');        
    else
        fid = fopen(['Simulation_Results/Belief_LLR_leaky_inhib_Delta_1_n_',num2str(n_exc),'_',num2str(n_inh),'_no_averaging_itrs_',num2str(no_averaging_itrs),'_w_rule_',num2str(weight_rule),'_binary_',num2str(binary_mode),'.txt'], 'a+');
    end
    fprintf(fid, 'T \t %d \t theta \t %f \t p \t %f \t q \t %f \t tau \t %f \t delta \t %f \t %f \t acc_plus \t %f \t acc_minus \t %f \t acc_0 \t %f \t',T,theta,p,q,tau,delta_plus,delta_minus,success_measure(1),success_measure(2),success_measure(3));
    fprintf(fid,'\n');
    fclose(fid);
end
%==========================================================================

