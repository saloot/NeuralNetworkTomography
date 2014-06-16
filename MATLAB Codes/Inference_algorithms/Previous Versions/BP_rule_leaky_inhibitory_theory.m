warning off

T = 500;
theta = .02;
Delta = 0.05;                            % Delta is the amount of update in each sample
delta_plus = 0.15;                      % the threshold on the belief that a weight is +1
delta_minus = 0.3;                     % the threshold on the belief that a weight is -1
epsilon_plus = .1;
epsilon_minus = .1;

p_plus = p * n_exc/n;
p_minus = p * n_inh/n;

p_ave = p_plus - p_minus;        
p_var = p_plus * (1-p_plus) + p_minus * (1-p_minus) + 2 *p_plus * p_minus;


P_c_plus = 0;
P_c_zero = 0;
P_c_minus = 0;


%===================CALCULATING AVERAGE TIME BETWEEN SPIKES================
t_fire = 1/(1-1/tau);
%==========================================================================


%==========================================================================
c_1 = (1-(1/tau^(t_fire + 1)) )/(1-(1/tau));
c_2 = (1-(1/tau^(2*t_fire + 2)) )/(1-(1/tau^2));

mu = (n-1)*q*p_ave*c_1;
sigma = sqrt((n-1)*q*p_var*c_2);

P_A_plus= q*(2-q);%q_function(delta_plus,q*c_1,sqrt(q*(1-q)*c_2));
P_A_minus = q;%q_function(delta_minus,q*c_1,sqrt(q*(1-q)*c_2));
%==========================================================================

%==========================CALCULATING P_c_plus============================
p_2_p = P_A_plus * q_function(theta * n-delta_plus,mu,sigma);
p_1_p = P_A_minus * (1-q_function(theta * n-delta_minus,mu,sigma));

p_2_m = P_A_plus * q_function(theta * n+delta_plus,mu,sigma);
p_1_m = P_A_minus * (1-q_function(theta * n+delta_minus,mu,sigma));

p_2_z = P_A_plus * q_function(theta * n,mu,sigma);
p_1_z = P_A_minus * (1-q_function(theta * n,mu,sigma));


% 
for l_2 = 0:T
    temp = 0;   
    temp_z = 0;        
    
    for l_1 = max(0,ceil(l_2 + epsilon_plus/Delta)):T-l_2                        
        temp = temp + nchoosek(T-l_2,l_1)*(p_2_p^l_1) * ((1-p_2_p)^(T-l_2-l_1));        
    end
        
    P_c_plus = P_c_plus + nchoosek(T,l_2)*(p_1_p^l_2) * ((1-p_1_p)^(T-l_2)) * temp;        
    
    if (l_2-epsilon_minus/Delta > T-l_2)
        temp_m =1;
    else
        temp_m = 0;
        for l_1 = 0:max(0,floor(l_2-epsilon_minus/Delta))
            temp_m = temp_m + nchoosek(T-l_2,l_1)*(p_2_m^l_1) * ((1-p_2_m)^(T-l_2-l_1));        
        end
    end
   
    P_c_minus = P_c_minus + nchoosek(T,l_2)*(p_1_m^l_2) * ((1-p_1_m)^(T-l_2)) * temp_m;
    
    for l_1 = max(0,ceil(l_2-epsilon_minus/Delta)):min(T-l_2,floor(l_2+epsilon/Delta))
        temp_z = temp_z + nchoosek(T-l_2,l_1)*(p_2_z^l_1) * ((1-p_2_z)^(T-l_2-l_1));
    end
    
    P_c_zero = P_c_zero + nchoosek(T,l_2)*(p_1_z^l_2) * ((1-p_1_z)^(T-l_2)) * temp_z;
    
    111;
end

%==========================================================================



%==========================================================================

P_c = P_c_plus * p_plus + P_c_minus * p_minus + P_c_zero * (1 - p_plus - p_minus);