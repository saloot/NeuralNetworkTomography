%==========================================================================
%***************FUNCTION: read_faulty_results_itr_3D***********************
%==========================================================================

%--------------------------------INPUTS------------------------------------
% N_in: The number of pattern nodes in the graph
% K_in: The dimension of the subspace of the pattern nodes
% L_in: The number of clusters if we have multiple levels (L = 1 for single level)
% alpha0: The step size in the learning algorithm
% beta0: The sparsity penalty coefficient in the learning algorithm
% theta0: The sparsity threshold in the learning algorithm
% gamma_BFO: The update threshold in the original bit-flipping recall algorithm 
% gamma_BFS: The update threshold in the simplified (yet better) bit-flipping recall algorithm 
% recall_algorithm_option: The recall algorithm identifier (0 for winner-take-all, 1 for the original bit flipping and 2 for the simplified bit flipping
% pattern_neur_noise: The maximum amount of noise a pattern neuron will "suffer" from
% const_neur_noise: The maximum amount of noise a constraint neuron will "suffer" from
%--------------------------------------------------------------------------

%--------------------------------OUTPUTS-----------------------------------
% processed_pattern_noise: The list of processed number of initial erroneous nodes
% processed_PER: The list of processed Pattern Error Rates
%--------------------------------------------------------------------------


%--------------------------FUNCTION DESCRIPTION----------------------------
% This function gets the specification of a faulty neural associative
% memory and reads the result of recall phase from the appropriate files. 
% The results will then be plotted and compared with theoretical values in
% a 3D manner.
%--------------------------------------------------------------------------

%==========================================================================
%==========================================================================




warning off


%%
%==============================INITIALIZATION==============================
addpath(genpath('/home1/amir/cluster/Common_Library')); 
figure_legend = [];                         % This variable will be field with the legend for the graphs

h_error_PER = figure;
% h_error_BER = figure;
set(01,'defaulttextinterpreter','tex')
%---------------------------Fixed Parameters-------------------------------

no_averaging_itrs =10;
n_exc = 160;
n_inh = 40;
n = n_exc + n_inh;
g = 2;
d_plus = p*n_exc/n;
d_minus = p*n_inh/n;
Delta0 = 10;
% check for delta plus and delta minus
theta = 0.02;
tau = 5.0153;
tau = 5.0168;
% tau = 5.0166;
% q = .85*(1-1/tau)*theta/(p_plus-p_minus);


T = 6000;
variable_parameter = 'q';
BP_flag = 1;
inhib_flag = 1;
weight_rule = 1;
B_LLR_flag = 0;
binary_mode = 3;
delta_plus = 0.5;
delta_minus = 0.9;
deg_min =2;
deg_max = 40;

q = 0.0600;

tau = 5.0171;
%--------------------------------------------------------------------------

    
%==========================================================================

%%
%============================PROCESS THE RESULTS===========================
%============================PROCESS THE RESULTS===========================
                   
%-----------------------------Read Recall Results--------------------------
if (B_LLR_flag)
    fid = fopen(['Simulation_Results/Power_law_Belief_LLR_leaky_inhib_n_',num2str(n_exc),'_',num2str(n_inh),...
                 '_no_averaging_itrs_',num2str(no_averaging_itrs),'_w_rule_',num2str(weight_rule),'_binary_',...
                  num2str(binary_mode),'_deg_',num2str(deg_min),'_',num2str(deg_max),'.txt'], 'r');
else        
    fid = fopen(['Simulation_Results/Power_law_Belief_LLR_leaky_inhib_Delta_1_n_',num2str(n_exc),'_',num2str(n_inh),...
                 '_no_averaging_itrs_',num2str(no_averaging_itrs),'_w_rule_',num2str(weight_rule),'_binary_',...
                  num2str(binary_mode),'_deg_',num2str(deg_min),'_',num2str(deg_max),'.txt'], 'r');
end

    
%Tdthetafgfdddqftaufdeltaffacc_plusfacc_minusfacc_0f
if (fid > -1)            
    results = fscanf(fid, '%s %d %s %f %s %f %s %d %d %s %f %s %f %s %f %f %s %f %s %f %s %f',[51,inf]);
    fclose(fid);       
else    
    error('Undefined input file!')   
end
%--------------------------------------------------------------------------        

%-----------------------Process the Results--------------------------------    
unprocessed_no_samples = results(2,:);    
unprocessed_theta = results(8,:);
unprocessed_g = results(10,:);
unprocessed_d_plus = results(12,:);
unprocessed_d_minus = results(13,:);
unprocessed_q = results(15,:);
unprocessed_tau = results(19,:);
unprocessed_delta_plus= results(25,:);
unprocessed_delta_minus= results(26,:);
unprocessed_acc_plus = results(35,:);
unprocessed_acc_minus = results(45,:);
unprocessed_acc_0 = results(51,:);
    



processed_no_samples = [];    
processed_theta = [];
processed_g = [];    
processed_parameter = [];
processed_tau = [];

   
unprocessed_parameter_1 = unique(unprocessed_d_plus);
unprocessed_parameter_2 = unique(unprocessed_d_minus);

l1 = length(unprocessed_parameter_1);
l2 = length(unprocessed_parameter_2);


processed_acc_plus = zeros(l1,l2);
processed_acc_minus = zeros(l1,l2);
processed_acc_zero = zeros(l1,l2);
processed_count = zeros(l1,l2);


for i = 1:length(unprocessed_no_samples)
    d_plus = unprocessed_d_plus(i);
    d_minus = unprocessed_d_minus(i);
    if (  (unprocessed_g(i) == g)  && (unprocessed_theta(i) == theta) && (norm(unprocessed_q(i)-q)<0.0001) && ( norm(unprocessed_tau(i) - tau) < 0.0001) && (unprocessed_delta_plus(i) == delta_plus) && (unprocessed_delta_minus(i) == delta_minus))
        kji = find(unprocessed_parameter_1 == d_plus);
        ihi = find(unprocessed_parameter_2 == d_minus);
    
        processed_acc_plus(kji,ihi) = processed_acc_plus(kji,ihi) + unprocessed_acc_plus(i);
        processed_acc_minus(kji,ihi) = processed_acc_minus(kji,ihi) + unprocessed_acc_minus(i);
        processed_acc_zero(kji,ihi) = processed_acc_zero(kji,ihi) + unprocessed_acc_0(i);
        processed_count(kji,ihi) = processed_count(kji,ihi) + 1;
    end
end
    
%==========================================================================

%==========================PLOT THE RESULTS================================
figure
processed_count = processed_count + 0.1* (processed_count == 0);
surf(unprocessed_parameter_1,unprocessed_parameter_2,(processed_acc_plus./processed_count)');
set(gca,'FontSize',24)
xlhand = get(gca,'xlabel');
ylhand = get(gca,'ylabel');
zlhand = get(gca,'zlabel');
set(xlhand,'string','$d_{+}$','fontsize',30)
set(ylhand,'string','$d_{-}$','fontsize',30)
set(zlhand,'string','$P_{C_+}$','fontsize',30)
colormap('Summer')


figure
surf(unprocessed_parameter_1,unprocessed_parameter_2,(processed_acc_minus./processed_count)');
set(gca,'FontSize',24)
xlhand = get(gca,'xlabel');
ylhand = get(gca,'ylabel');
zlhand = get(gca,'zlabel');
set(xlhand,'string','$d_{+}$','fontsize',30)
set(ylhand,'string','$d_{-}$','fontsize',30)
set(zlhand,'string','$P_{C_-}$','fontsize',30)
colormap('Autumn')

figure
surf(unprocessed_parameter_1,unprocessed_parameter_2,(processed_acc_zero./processed_count)');
set(gca,'FontSize',24)
xlhand = get(gca,'xlabel');
ylhand = get(gca,'ylabel');
zlhand = get(gca,'zlabel');
set(xlhand,'string','$d_{+}$','fontsize',30)
set(ylhand,'string','$d_{-}$','fontsize',30)
set(zlhand,'string','$P_{C_0}$','fontsize',30)
colormap('Winter')

acc_p = processed_acc_plus./processed_count;
acc_m = processed_acc_minus./processed_count;
acc_0 = processed_acc_zero./processed_count;
acc = zeros(l1,l2);
for i = 1:l1
    for j = 1:l2
        acc(i,j) = unprocessed_parameter_1(i) * acc_p(i,j)/n + unprocessed_parameter_2(j) * acc_m(i,j)/n + (1-(unprocessed_parameter_1(i)+unprocessed_parameter_2(j))/n) * acc_0(i,j);
    end

end

figure

surf(unprocessed_parameter_1,unprocessed_parameter_2,acc');
set(gca,'FontSize',24)
xlhand = get(gca,'xlabel');
ylhand = get(gca,'ylabel');
zlhand = get(gca,'zlabel');
set(xlhand,'string','$d_{+}$','fontsize',30)
set(ylhand,'string','$d_{-}$','fontsize',30)
set(zlhand,'string','$P_{C}$','fontsize',30)

%==========================================================================


%======================ADD THE LEGEND TO THE FIGURES=======================

% figure(h_error_PER)
% legend(figure_legend)


111;
% figure(h_error_BER_BFO)
% legend(figure_legend)
% figure(h_error_BER_WTA)
% legend(figure_legend)
%==========================================================================

