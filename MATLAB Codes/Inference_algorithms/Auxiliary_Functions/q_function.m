function Q = q_function(x,mu,sigma)
temp = (x-mu)/sigma;
Q = 1-erf(temp/sqrt(2));
Q = Q/2;
