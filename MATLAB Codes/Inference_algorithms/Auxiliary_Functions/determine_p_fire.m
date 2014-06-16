function [p1] = determine_p_fire(n,p,q,theta,itr_max)
warning off 

% q = theta/p;
%===========================INITIALIZATION=================================
avg_no_edge = round(p*n);       % estimated number of edges in the graph
no_edges_fire = ceil(theta*n);  % estimated number of edges required for a neuron to fire
%==========================================================================


%===============PROBABILITY OF A NEURON FIRING OVER TIME===================
for itr = 1:itr_max
    p1 = 0;
    for j = no_edges_fire:floor(avg_no_edge)
        p1 = p1 + nchoosek(avg_no_edge,j) * (q^j) * ((1-q)^(avg_no_edge-j));
    end
    if (norm(q-p1)/q < .0001)
        break;
    else
        q = p1;
    end
end
%==========================================================================
