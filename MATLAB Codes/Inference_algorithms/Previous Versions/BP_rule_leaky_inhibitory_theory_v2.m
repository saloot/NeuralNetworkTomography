function [P_C,P_C_plus,P_C_0,P_C_minus,P_E] = BP_rule_leaky_inhibitory_theory_v2(n,tau,theta,T,p_plus,p_minus,q,delta_plus,delta_minus,Delta)


%=============================INITIALIZATION===============================
t_fire = 1/(1-1/tau);                           % The average weighting time between spikes
c_1 = (1-(1/tau^(t_fire + 1)) )/(1-(1/tau));
c_2 = (1-(1/tau^(2*t_fire + 2)) )/(1-(1/tau^2));

% P_A_plus = q_function(delta_plus,q*c_1,sqrt(q*(1-q)*c_2));            % probability that the net input from a particular neuron is larger than delta_plus
% P_A_minus = q_function(delta_minus,q*c_1,sqrt(q*(1-q)*c_2));          % probability that the net input from a particular neuron is larger than delta_minus
P_A_plus = q;                                   % probability that the net input from a particular neuron is larger than delta_plus
P_A_minus = q;                                  % probability that the net input from a particular neuron is larger than delta_minus
c_1 = 1;
c_2 = 1;
%==========================================================================



%========CALCULATING PROBABILITY OF INCREASING/DECREASING A WEIGHT=========
mu = c_1*(n-1)*q*(p_plus - p_minus);                                  
sigma = sqrt(c_2*(n-1)*q*(p_plus*(1-q*p_plus)+p_minus*(1-q*p_minus)));

p_inc_p = P_A_plus*q_function(theta * n - delta_plus,mu,sigma);             % Probability of increasing the weight on the condition that Gi = 1
p_inc_m = P_A_plus*q_function(theta * n + delta_plus,mu,sigma);             % Probability of increasing the weight on the condition that Gi = -1
p_inc_0 = P_A_plus*q_function(theta * n ,mu,sigma);                         % Probability of increasing the weight on the condition that Gi = 0

p_dec_p = P_A_minus*(1-q_function(theta * n - delta_minus,mu,sigma));       % Probability of decreasing the weight on the condition that Gi = 1
p_dec_m = P_A_minus*(1-q_function(theta * n + delta_minus,mu,sigma));       % Probability of decreasing the weight on the condition that Gi = -1
p_dec_0 = P_A_minus*(1-q_function(theta * n ,mu,sigma));                    % Probability of decreasing the weight on the condition that Gi = 0
%==========================================================================


%======CALCULATE MEAN AND VARIANCE OF THE CONDITIONAL RANDOM VARIABLES=====
mu1 = Delta*T*(p_inc_p-p_dec_p);
var1 = Delta*sqrt(T*(p_inc_p*(1-p_inc_p) + p_dec_p*(1-p_dec_p)));

mu2 = Delta*T*(p_inc_m-p_dec_m);
var2 = Delta*sqrt(T*(p_inc_m*(1-p_inc_m) + p_dec_m*(1-p_dec_m)));

mu3 = Delta*T*(p_inc_0-p_dec_0);
var3 = Delta*sqrt(T*(p_inc_0*(1-p_inc_0) + p_dec_0*(1-p_dec_0)));

epsilon_plus = mu1-(mu1-mu3)/4;%(mu1+mu3)/2;
epsilon_minus = abs(mu2 + (mu3-mu2)/4);
% epsilon_plus = (mu1+mu3)/2;
% epsilon_minus = abs(mu2+mu3)/2;
%==========================================================================

%==============CALCULATE PROBABILITY OF DECLARING A WEIGH +1==============
P_C_plus = q_function(epsilon_plus,mu1,var1);
P_E_plus = (1-p_plus - p_minus) * q_function(epsilon_plus,mu3,var3)+p_minus * q_function(epsilon_plus,mu2,var2);
%==========================================================================


%==============CALCULATE PROBABILITY OF DECLARING A WEIGH -1==============
P_C_minus = 1-q_function(-epsilon_minus,mu2,var2);
P_E_minus = (1-p_plus - p_minus) * (1-q_function(-epsilon_minus,mu3,var3))+p_plus * (1-q_function(-epsilon_minus,mu1,var1));
%==========================================================================

%==============CALCULATE PROBABILITY OF DECLARING A WEIGH 0===============
P_C_0 = q_function(-epsilon_minus,mu3,var3)-q_function(epsilon_plus,mu3,var3);
P_E_0 = p_minus * (q_function(-epsilon_minus,mu2,var2)-q_function(epsilon_plus,mu2,var2))+p_plus * (q_function(-epsilon_minus,mu1,var1)-q_function(epsilon_plus,mu1,var1));
%==========================================================================


%=================OVERALL PROBABAILITY OF SUCCESS==========================
P_C = p_plus * P_C_plus + p_minus * P_C_minus + (1-p_plus-p_minus) * P_C_0;
P_E = P_E_plus + P_E_minus + P_E_0;
%==========================================================================
