function B_LLR = calculate_belief_LLR(no_edges_fire,L,p,R)

if (R)
    P1 = 0;
    for j = no_edges_fire:L-1
        P1 = P1 + nchoosek(L-1,j) * (p^j) * ( (1-p)^(L-1-j) );
    end

    P2 = P1;
    j = no_edges_fire-1;
    P1 = P2 + nchoosek(L-1,j)*(p^j) * ( (1-p)^(L-1-j) );

    B_LLR = log(P1/P2);
else
    P1 = 0;
    for j = 0:no_edges_fire-2
        P1 = P1 + nchoosek(L-1,j) * (p^j) * ( (1-p)^(L-1-j) );
    end

    P2 = P1;
    j = no_edges_fire-1;
    P1 = P2 + nchoosek(L-1,j)*(p^j) * ( (1-p)^(L-1-j) );

    B_LLR = log(P1/P2);
end
    
