function [W,epsilon_plus_opt,epsilon_minus_opt,d_plus_min,d_minus_min] = epsilon_finder_powerlaw(W2,T,n_exc,n_inh,q,theta,tau,delta_plus,delta_minus,Delta,deg_min,deg_max,aa,r,binary_mode,weight_rule)
warning off
n = n_exc + n_inh;
min_norm_lus = Inf;
min_norm_minus = Inf;
min_norm = Inf;
nor = [];
for d = 1:deg_max  

   if (d<deg_max)
   d_plus = round(d* n_exc/n);
   for d_minus = max(0,d-d_plus-1):max(0,d-d_plus+1)          
    % P_A_plus = q_function(B_LLR_thr,q*c_1,sqrt(q*(1-q)*c_2));            % probability that the net input from a particular neuron is larger than B_LLR_thr
    % P_A_minus = q_function(B_LLR_thr,q*c_1,sqrt(q*(1-q)*c_2));          % probability that the net input from a particular neuron is larger than B_LLR_thr
    P_A_plus = q;                                   % probability that the net input from a particular neuron is larger than B_LLR_thr
    P_A_minus = q;                                  % probability that the net input from a particular neuron is larger than B_LLR_thr
    c_1 = 1;
    c_2 = 1;

    
    x_min = -max(5,2*max(d_plus,d_minus));
    x_max = max(5,2*max(d_plus,d_minus));
    T_fire = 1;
    T_fire = round(1/r-1);
    if (d_plus == 1)
        p_inc_p = P_A_plus;            % Probability of increasing the weight on the condition that Gi = 1
    else
        p_inc_p = P_A_plus* calculate_p_plus(d_plus,d_minus,q,x_min,x_max,theta,tau,T_fire,delta_plus);
    end    
    if (d_minus == 1)
        p_inc_m = 0;
    else
        p_inc_m = P_A_plus*calculate_p_minus(d_plus,d_minus,q,x_min,x_max,theta,tau,T_fire,delta_plus);             % Probability of increasing the weight on the condition that Gi = -1
    end
    
    p_inc_0 = P_A_plus*calculate_p_zero(d_plus,d_minus,q,x_min,x_max,theta,tau,T_fire,delta_plus);                         % Probability of increasing the weight on the condition that Gi = 0
    
    if (d_plus == 1)
        p_dec_p = 0;
    else
        p_dec_p = P_A_minus*(1-calculate_p_plus(d_plus,d_minus,q,x_min,x_max,theta,tau,T_fire,delta_minus));       % Probability of decreasing the weight on the condition that Gi = 1
    end
    
    if (d_minus == 1)
        p_dec_m = P_A_minus;
    else    
        p_dec_m = P_A_minus*(1-calculate_p_minus(d_plus,d_minus,q,x_min,x_max,theta,tau,T_fire,delta_minus));       % Probability of decreasing the weight on the condition that Gi = -1
    end
    p_dec_0 = P_A_minus*(1-calculate_p_zero(d_plus,d_minus,q,x_min,x_max,theta,tau,T_fire,delta_plus));                    % Probability of decreasing the weight on the condition that Gi = 0


    
    mu1 = Delta*T*(p_inc_p-p_dec_p);
    var1 = Delta*sqrt(T*(p_inc_p*(1-p_inc_p) + p_dec_p*(1-p_dec_p)));

    mu2 = Delta*T*(p_inc_m-p_dec_m);
    var2 = Delta*sqrt(T*(p_inc_m*(1-p_inc_m) + p_dec_m*(1-p_dec_m)));

    mu3 = Delta*T*(p_inc_0-p_dec_0);
    var3 = Delta*sqrt(T*(p_inc_0*(1-p_inc_0) + p_dec_0*(1-p_dec_0)));

    if (weight_rule == 1)
        epsilon_plus = mu1-(mu1-mu3)/2;%(mu1+mu3)/2;
        epsilon_minus = mu2 + (mu3-mu2)/5;
    else
        a = 0.5*(1+Delta)^(mu1/Delta);
        b = 0.5*(1+Delta)^(mu2/Delta);
        c = 0.5*(1+Delta)^(mu3/Delta);
        epsilon_plus = (a+c)/2;
        epsilon_minus = (b+c)/2;
    end
    
    
    if (weight_rule == 1)        
        epsilon_plus_modified = epsilon_plus * aa*((2*r)-1)/(q*(2*p_inc_0/q-1));
        epsilon_minus_modified = epsilon_minus * aa*((2*r)-1)/(q*(2*p_inc_0/q-1));
    else        
        epsilon_plus_modified = epsilon_plus * (1+Delta)^(T*(aa*((2*r)-1)-q*(2*p_inc_0-1)));     
        epsilon_minus_modified = epsilon_minus * (1+Delta)^(T*(aa*((2*r)-1)-q*(2*p_inc_0-1)));
    end
    
    if (binary_mode == 1)
        epsilon_plus_modified = epsilon_plus;
        epsilon_minus_modified = epsilon_minus;
        W = (W2>epsilon_plus)-(W2<-epsilon_minus);
    else
        W = (W2>epsilon_plus_modified)-(W2<epsilon_minus_modified);
    end
    
    E1 = sum(W>0);
    E2 = sum(W<0);
%     if ( norm(d_plus - E1)+norm(d_minus - E2) < min_norm)
    nor = [nor,norm(r-p_inc_0/q)];
    if (norm(r-p_inc_0/q) < min_norm)
        d_plus_min = d_plus;
        d_minus_min = d_minus;
        min_norm = norm(d_plus - E1)+norm(d_minus - E2);
min_norm = norm(r-p_inc_0/q);
        epsilon_plus_opt= epsilon_plus_modified;
        epsilon_minus_opt = epsilon_minus_modified;
    end
   end

else
    P_A_plus = q;                                   % probability that the net input from a particular neuron is larger than B_LLR_thr
    P_A_minus = q;                                  % probability that the net input from a particular neuron is larger than B_LLR_thr
    c_1 = 1;
    c_2 = 1;
    d_plus = round(d* n_exc/n);
    d_minus = d - d_plus;
    p_plus = d_plus/n;
    p_minus = d_minus/n;
    p = p_minus + p_plus;

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
        epsilon_plus = mu1-(mu1-mu3)/5;%(mu1+mu3)/2;
        epsilon_minus = mu2 + (mu3-mu2)/5;
    else
        a = 0.5*(1+Delta)^(mu1/Delta);
        b = 0.5*(1+Delta)^(mu2/Delta);
        c = 0.5*(1+Delta)^(mu3/Delta);
        epsilon_plus = (a+c)/2;
        epsilon_minus = (b+c)/2;
    end
    
    
    if (weight_rule == 1)        
        epsilon_plus_modified = epsilon_plus * aa*((2*r)-1)/(q*(2*p_inc_0/q-1));
        epsilon_minus_modified = epsilon_minus * aa*((2*r)-1)/(q*(2*p_inc_0/q-1));
    else        
        epsilon_plus_modified = epsilon_plus * (1+Delta)^(T*(aa*((2*r)-1)-q*(2*p_inc_0-1)));     
        epsilon_minus_modified = epsilon_minus * (1+Delta)^(T*(aa*((2*r)-1)-q*(2*p_inc_0-1)));
    end
    
    if (binary_mode == 1)
        epsilon_plus_modified = epsilon_plus;
        epsilon_minus_modified = epsilon_minus;
        W = (W2>epsilon_plus)-(W2<-epsilon_minus);
    else
        W = (W2>epsilon_plus_modified)-(W2<epsilon_minus_modified);
    end
    
    E1 = sum(W>0);
    E2 = sum(W<0);
%     if ( norm(d_plus - E1)+norm(d_minus - E2) < min_norm)
    nor = [nor,norm(r-p_inc_0/q)];
    if (norm(r-p_inc_0/q) < min_norm)
        d_plus_min = d_plus;
        d_minus_min = d_minus;
        min_norm = norm(d_plus - E1)+norm(d_minus - E2);
        min_norm = norm(r-p_inc_0/q);
        epsilon_plus_opt= epsilon_plus_modified;
        epsilon_minus_opt = epsilon_minus_modified;
    end

end
    
end

W = (W2>epsilon_plus_opt)-(W2<-epsilon_minus_opt);
[W_sorted,ind] = sort(W2);
W = zeros(1,n);
W(ind(end-d_plus_min+1:end)) = 1;
W(ind(1:d_minus_min)) = -1;

