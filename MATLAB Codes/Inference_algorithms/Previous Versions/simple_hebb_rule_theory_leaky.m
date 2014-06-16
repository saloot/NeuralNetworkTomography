function [p1_T,p2_T] = simple_hebb_rule_theory_leaky(n,p,q,no_samples,theta,tau,avg_T_fire)
warning off 
% parameter_range = [.0:.01:.5];
% p1_total = zeros(1,length(parameter_range));
% p2_total = zeros(1,length(parameter_range));
% 
% for itr = 1:length(parameter_range)
%     q = parameter_range(itr);

%===========================INITIALIZATION=================================
% p = 0.4;                        % connection probability
% n = 100;                        % number of neurons
% no_samples = 1000;                        % number of sample recordings
% theta = .05;                     % update threshold

% q = 0.15;                        % stimulus firing probability

avg_no_edge = round(p*n);       % estimated number of edges in the graph
no_edges_fire = ceil(theta*n);  % estimated number of edges required for a neuron to fire
%==========================================================================



%==========PROBABILITY OF CORRECTLY GUESSING ONE LINK IN ONE TRIAL=========
temp = no_edges_fire - (avg_no_edge-1) *q * (1-1/(tau^avg_T_fire))/(1-1/tau);
temp = temp/sqrt((avg_no_edge-1) *q * (1-q) * (1-1/(tau^(2*avg_T_fire)))/(1-1/(tau^2)) );
p1 = q * (1-erf(temp/sqrt(2)))/2;
p3 = 1-p1;
%==========================================================================


%=========PROBABILITY OF INCORRECTLY GUESSING ONE LINK IN ONE TRIAL========
temp = no_edges_fire - avg_no_edge *q * (1-1/(tau^avg_T_fire))/(1-1/tau);
temp = temp/sqrt(avg_no_edge *q * (1-q) * (1-1/(tau^(2*avg_T_fire)))/(1-1/(tau^2)) );
p2 = q * (1-erf(temp/sqrt(2)))/2;
%==========================================================================

w_thr = (p1+p2)/2;

%===PROBABILITY OF CORRECTLY & INCORRECTLY GUESSING ONE LINK IN no_samples TRIALS===
p1_T = 0;
p2_T = 0;
p3_T = 0;
for j = 1:floor(w_thr*no_samples)
    p1_T = p1_T + nchoosek(no_samples,j) * (p1^j) * ( (1-p1)^(no_samples-j) );
    p2_T = p2_T + nchoosek(no_samples,j) * (p2^j) * ( (1-p2)^(no_samples-j) );
    p3_T = p3_T + nchoosek(no_samples,j) * (p3^j) * ( (1-p3)^(no_samples-j) );
end
p1_T = 1-p1_T;
p2_T = 1-p2_T+p3_T;
%==========================================================================


% %=========PROBABILITY OF INCORRECTLY GUESSING ONE LINK IN ONE TRIAL========
% p1_total(itr) = p1_T;
% p2_total(itr) = p2_T;
% end
% 
% 
% figure
% plot(parameter_range,p1_total,'b')
% hold on
% plot(parameter_range,p2_total,'r')