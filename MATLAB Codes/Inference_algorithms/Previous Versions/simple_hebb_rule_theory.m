function [p1_T,p2_T] = simple_hebb_rule_theory(n,p,q,T,theta,hist_window)
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
% T = 1000;                        % number of sample recordings
% theta = .05;                     % update threshold

% q = 0.15;                        % stimulus firing probability

avg_no_edge = round(p*n);       % estimated number of edges in the graph
no_edges_fire = ceil(theta*n);  % estimated number of edges required for a neuron to fire
%==========================================================================



%==========PROBABILITY OF CORRECTLY GUESSING ONE LINK IN ONE TRIAL=========
p1 = 0;
for j = no_edges_fire-1:floor(avg_no_edge)
    if (hist_window == 0)
        p1 = p1 + nchoosek(avg_no_edge,j) * (q^j) * ((1-q)^(avg_no_edge-j));
    else
        p1 = p1 + nchoosek(avg_no_edge,j) * (q^j) * ((1-q)^(avg_no_edge-j))*(1-q)^(hist_window);
    end
end
p1 = q*p1;
p3 = 1-p1;
%==========================================================================


%=========PROBABILITY OF INCORRECTLY GUESSING ONE LINK IN ONE TRIAL========
if (no_edges_fire < avg_no_edge)
    if (hist_window == 0)
        p2 = p1 -q*nchoosek(avg_no_edge,no_edges_fire-1) * (q^(no_edges_fire-1)) * ((1-q)^(avg_no_edge-(no_edges_fire-1)));
    else
        p2 = p1 -q*nchoosek(avg_no_edge,no_edges_fire-1) * (q^(no_edges_fire-1)) * ((1-q)^(avg_no_edge-(no_edges_fire-1)))*(1-q)^(hist_window);
    end
else
    p2 = p1;
end
%==========================================================================

w_thr = (p1+p2)/2;

%===PROBABILITY OF CORRECTLY & INCORRECTLY GUESSING ONE LINK IN T TRIALS===
p1_T = 0;
p2_T = 0;
p3_T = 0;
for j = 1:floor(w_thr*T)
    p1_T = p1_T + nchoosek(T,j) * (p1^j) * ( (1-p1)^(T-j) );
    p2_T = p2_T + nchoosek(T,j) * (p2^j) * ( (1-p2)^(T-j) );
    p3_T = p3_T + nchoosek(T,j) * (p3^j) * ( (1-p3)^(T-j) );
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