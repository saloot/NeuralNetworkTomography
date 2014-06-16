function G = power_law_inhibitory(m,n_exc,n_inh,deg_min,deg_max,gamma)

%=============================INITIALIZATIONS==============================
n = n_exc + n_inh;
deg = zeros(1,m);
deg_exponents = [deg_min:deg_max].^(-gamma);
deg_exponents = deg_exponents/sum(deg_exponents);
%==========================================================================

%=========================DETERMINE DEGREE DISTRIBUTION====================
for i = 1:m
    a = rand;   
    temp = 0;
    for j = 1:length(deg_exponents)
        temp = temp + deg_exponents(j);
        if (a < temp)
            k = deg_min+j-1;
            break;        
        end
    end
        
    deg(i) = k;
end
%==========================================================================

%-------------------------Vectorized Version-------------------------------
G = [];
for i = 1:m
    d_plus = round(deg(i) *n_exc/n);
    d_minus = deg(i) - d_plus;
    temp = zeros(1,n);
    ind = randperm(n_exc);
    
    temp(ind(1:d_plus)) = 1;
    
    ind = randperm(n_inh);
    if (d_minus > 0)
        temp(n_exc + ind(1:d_minus)) = -1;
    end    
    G = [G,temp];    
    
end
%--------------------------------------------------------------------------


if (m==n_exc+n_inh)
    for i = 1:min(m,n_exc+n_inh)
        G(i,i) = 0;
    end
end