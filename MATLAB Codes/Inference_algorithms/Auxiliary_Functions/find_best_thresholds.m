T = 500;

epsilon_plus_range = [0.0001:.01:.2];
epsilon_minus_range = [0.0001:.01:.2];
param_range_1 = [0.01:.05:.4];
param_range_2 = [500:500:2000];
% q = .5*theta/(p_plus-p_minus);



P_C_p_tot = zeros(length(param_range_1),length(param_range_2));
P_C_m_tot = zeros(length(param_range_1),length(param_range_2));
P_C_tot = zeros(length(param_range_1),length(param_range_2));

for itr1 = 1:length(param_range_1)
    q = param_range_1(itr1);
    for itr2 = 1:length(param_range_2)
        T = param_range_2(itr2);        
        [P_C,P_C_plus,P_C_minus,P_E] = BP_rule_leaky_inhibitory_theory_v2(n,tau,theta,T,p_plus,p_minus,q,delta_plus,delta_minus,Delta);
        P_C_p_tot(itr1,itr2) = P_C_plus;
        P_C_m_tot(itr1,itr2) = P_C_minus;
        P_C_tot(itr1,itr2) = P_C;
    end
end

figure
surf(param_range_2,param_range_1,P_C_p_tot);
hold on 
surf(param_range_2,param_range_1,P_C_m_tot,'y');
ylabel('param_1')
xlabel('param_2')
legend('P_{C_+}','P_{C_-}')