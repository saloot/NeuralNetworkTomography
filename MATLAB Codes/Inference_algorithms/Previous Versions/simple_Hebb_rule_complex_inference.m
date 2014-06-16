W = zeros(1,n);
fixed_weights = zeros(1,n);
% S = R(:,1:n);
% R = R(:,n+1);
end_flag = 0;
while (end_flag==0)
    ss = sum(S');
    if sum(ss/(theta*n)
    for i = 1:length(ss)
        if ( (ss(i) == theta*n) && (R(i) == 1) )
            for j = 1:n
                if (S(i,j) == 1)
                    W(j) = 1;
                    fixed_weights(j) = 1;
                end
            end
        end
    end
end