beta = 1.15;
epsilon = 0.1;
epsilon_2 = epsilon;
gamma = log(1+2*epsilon)/log(beta);
gamma_prime = log(1-2*epsilon_2)/log(beta);
t_fire = 0;
p = .3;

n_exc = 240;
n_inh = 60;
n = n_exc + n_inh;

p_plus = p * n_exc / n;
p_minus = p * n_inh / n;
% p_plus 
% p_minus
p_var = p_plus * (1-p_plus) + p_minus * (1-p_minus) + 2 *p_plus * p_minus;
% p_var = p_plus *(1+p_plus/p_minus);
thr_fire = theta*n;
q = 1.0*theta/(p_plus-p_minus);%+1/(n-n/tau));

mu_plus = q * n * (p_plus - p_minus);
sigma_plus = sqrt(q*n*p_var);
prob_not_fire_so_far = 1;




for i = 1:T
    tau_sum1 = (1-(1/tau)^i)/(1-(1/tau));
    tau_sum2 = sqrt((1-(1/tau^2)^i)/(1-(1/tau^2)));    
    prob_fire = q_function(thr_fire,mu_plus*tau_sum1,sigma_plus*tau_sum2);
    
    t_fire = t_fire + i *prob_fire *prob_not_fire_so_far;
    prob_not_fire_so_far = prob_not_fire_so_far * (1-prob_fire);
    
    if (norm(prob_fire *prob_not_fire_so_far)<.00001)
        break;
    end
end

qq = 1-(1-q)^t_fire;
T = 500;
delta = q * (1-(1/tau)^t_fire)/(1-(1/tau));
delta = 0.1;
Q_plus = q_function(thr_fire-delta,mu_plus,sigma_plus);
Q_plus_1 = q_function(thr_fire,mu_plus,sigma_plus);
Q_plus_2 = q_function(thr_fire+delta,mu_plus,sigma_plus);

mu_A = t_fire*q *(1-(1/tau)^t_fire)/(1-(1/tau));
sigma_A = sqrt(t_fire*q *(1-q) *(1-(1/tau^2)^t_fire)/(1-(1/tau^2)));
qq = q_function(delta,mu_A,sigma_A);


Q_plus-0.5-(gamma/2/qq/T)

0.5+(gamma/2/qq/T)-Q_plus_1
Q_plus_1 - 0.5-(gamma_prime/2/qq/T)

 0.5+(gamma_prime/2/qq/T)-Q_plus_2


% T_lower = gamma/(2*qq)/(Q_plus-0.5)


