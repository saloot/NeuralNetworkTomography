function [w_thr] = determine_weight_thr(n,p,q,theta)
warning off 

%===========================INITIALIZATION=================================
avg_no_edge = round(p*n);       % estimated number of edges in the graph
no_edges_fire = ceil(theta*n);  % estimated number of edges required for a neuron to fire
%==========================================================================



%==========PROBABILITY OF CORRECTLY GUESSING ONE LINK IN ONE TRIAL=========
p1 = 0;
for j = no_edges_fire-1:floor(avg_no_edge)
    p1 = p1 + nchoosek(avg_no_edge,j) * (q^j) * ((1-q)^(avg_no_edge-j));
end
p1 = q*p1;
%==========================================================================


%=========PROBABILITY OF INCORRECTLY GUESSING ONE LINK IN ONE TRIAL========
p2 = p1 -q*nchoosek(avg_no_edge,no_edges_fire-1) * (q^(no_edges_fire-1)) * ((1-q)^(avg_no_edge-(no_edges_fire-1)));
%==========================================================================

w_thr = (p1+p2)/2;