function [W,epsilon_plus_opt,epsilon_minus_opt] = epsilon_finder(W2,T,n_exc,n_inh,q,theta,delta_plus,delta_minus,Delta,deg_min,deg_max,aa,r,binary_mode,weight_rule)

n = n_exc + n_inh;
min_norm_plus = Inf;
min_norm_minus = Inf;
min_norm = Inf;

for d = 1:deg_max
   p = d/n; 
   p_plus = p* n_exc/n;
   p_minus = p * n_inh/n;            
    % P_A_plus = q_function(B_LLR_thr,q*c_1,sqrt(q*(1-q)*c_2));            % probability that the net input from a particular neuron is larger than B_LLR_thr
    % P_A_minus = q_function(B_LLR_thr,q*c_1,sqrt(q*(1-q)*c_2));          % probability that the net input from a particular neuron is larger than B_LLR_thr
    P_A_plus = q;                                   % probability that the net input from a particular neuron is larger than B_LLR_thr
    P_A_minus = q;                                  % probability that the net input from a particular neuron is larger than B_LLR_thr
    c_1 = 1;
    c_2 = 1;

    mu = c_1*(n-1)*q*(p_plus - p_minus);                                  
    sigma = sqrt(c_2*(n-1)*q*(p_plus*(1-q*p_plus)+p_minus*(1-q*p_minus)));

    p_inc_p = P_A_plus*q_function(theta * p * n - delta_plus,mu,sigma);             % Probability of increasing the weight on the condition that Gi = 1
    p_inc_m = P_A_plus*q_function(theta * p * n + delta_plus,mu,sigma);             % Probability of increasing the weight on the condition that Gi = -1
    p_inc_0 = P_A_plus*q_function(theta * p * n ,mu,sigma);                         % Probability of increasing the weight on the condition that Gi = 0

    p_dec_p = P_A_minus*(1-q_function(theta * p * n - delta_minus,mu,sigma));       % Probability of decreasing the weight on the condition that Gi = 1
    p_dec_m = P_A_minus*(1-q_function(theta * p * n + delta_minus,mu,sigma));       % Probability of decreasing the weight on the condition that Gi = -1
    p_dec_0 = P_A_minus*(1-q_function(theta * p * n ,mu,sigma));                    % Probability of decreasing the weight on the condition that Gi = 0


    
    mu1 = Delta*T*(p_inc_p-p_dec_p);
    var1 = Delta*sqrt(T*(p_inc_p*(1-p_inc_p) + p_dec_p*(1-p_dec_p)));

    mu2 = Delta*T*(p_inc_m-p_dec_m);
    var2 = Delta*sqrt(T*(p_inc_m*(1-p_inc_m) + p_dec_m*(1-p_dec_m)));

    mu3 = Delta*T*(p_inc_0-p_dec_0);
    var3 = Delta*sqrt(T*(p_inc_0*(1-p_inc_0) + p_dec_0*(1-p_dec_0)));

    if (weight_rule == 1)
        epsilon_plus = mu1-(mu1-mu3)/4;%(mu1+mu3)/2;
        epsilon_minus = abs(mu2 + (mu3-mu2)/4);
    else
        a = 0.5*(1+Delta)^(mu1/Delta);
        b = 0.5*(1+Delta)^(mu2/Delta);
        c = 0.5*(1+Delta)^(mu3/Delta);
        epsilon_plus = (a+c)/2;
        epsilon_minus = (b+c)/2;
    end
    
    muu = (n-1)*q*(p_plus - p_minus);                                  
    sigmaa = sqrt((n-1)*q*(p_plus*(1-q*p_plus)+p_minus*(1-q*p_minus)));
        
    if (weight_rule == 1)        
        epsilon_plus_modified = epsilon_plus * aa*((2*r)-1)/(q*(2*q_function(theta * p * n ,muu,sigmaa)-1));
        epsilon_minus_modified = epsilon_minus * aa*((2*r)-1)/(q*(2*q_function(theta * p *  n ,muu,sigmaa)-1));
    else        
        epsilon_plus_modified = epsilon_plus * (1+Delta)^(T*(aa*((2*r)-1)-q*(2*q_function(theta * p * n ,muu,sigmaa)-1)));     
        epsilon_minus_modified = epsilon_minus * (1+Delta)^(T*(aa*((2*r)-1)-q*(2*q_function(theta * p * n ,muu,sigmaa)-1)));
    end
    
    if (binary_mode == 1)
        W = (W2>epsilon_plus)-(W2<-epsilon_minus);
    else
        W = (W2>epsilon_plus_modified)-(W2<-epsilon_minus_modified);
    end
    
    E1 = sum(W>0);
    E2 = sum(W<0);
    if ( norm(p_plus*n - E1)+norm(p_minus*n - E2) < min_norm)
        p_min = p;
        min_norm = norm(p_plus*n - E1)+norm(p_minus*n - E2);
        epsilon_plus_opt= epsilon_plus_modified;
        epsilon_minus_opt = epsilon_minus_modified;
    end
    
end

W = (W2>epsilon_plus_opt)-(W2<-epsilon_minus_opt);


%      % w_thr = determine_weight_thr_leaky(n,p,q,theta);
%      % w_thr = q*(1/avg_T_fire+1/avg_T_fire2)/2;
%      % W = (W>w_thr);        
%      [W_sorted,ind] = sort(W);
%     W = zeros(1,n);
%     WW = zeros(1,n);     
% %      W(ind(end-sum(G>0)+1:end)) = abs(sign(W_sorted(end-sum(G>0)+1:end)));        
% %      W(ind(1:sum(G<0))) = -abs(sign(W_sorted(1:sum(G<0))));            
%     
%     muu = (n-1)*q*(p_plus - p_minus);                                  
%     sigmaa = sqrt((n-1)*q*(p_plus*(1-q*p_plus)+p_minus*(1-q*p_minus)));
%     
%     aa = sum(sum(S))/T/n;
%     if (weight_rule == 1)        
%         epsilon_plus_modified = epsilon_plus * aa*((2*sum(R)/T)-1)/(q*(2*q_function(theta * n ,muu,sigmaa)-1));
%         epsilon_minus_modified = epsilon_minus * aa*((2*sum(R)/T)-1)/(q*(2*q_function(theta * n ,muu,sigmaa)-1));
%     else        
%         epsilon_plus_modified = epsilon_plus * (1+Delta)^(T*(aa*((2*sum(R)/T)-1)-q*(2*q_function(theta * n ,muu,sigmaa)-1)));     
%         epsilon_minus_modified = epsilon_minus * (1+Delta)^(T*(aa*((2*sum(R)/T)-1)-q*(2*q_function(theta * n ,muu,sigmaa)-1)));
%     end
%      
%     if (binary_mode == 1)                    
%         W = (W2>epsilon_plus)-(W2<-epsilon_minus);
%     elseif (binary_mode == 2)
%        W = (W2>epsilon_plus_modified)-(W2<-epsilon_minus_modified);
%     else
%         
%         W = zeros(1,n);        
%         W(ind(end-round(p_plus*n)+1:end)) = 1;
%         W(ind(1:round(p_minus*n))) = -1;
%     end