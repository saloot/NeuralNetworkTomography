val = [];
ind = [];
for L = no_edges_fire:no_edges_fire+20
    ind = [ind,L];   
%     val = [val,nchoosek(L-1,no_edges_fire-1)*(p^(no_edges_fire-1))* ( (1-p)^(L-no_edges_fire))];    
    val = [val,calculate_belief_LLR(no_edges_fire,L,p,0)];
end
figure
plot(ind,val)