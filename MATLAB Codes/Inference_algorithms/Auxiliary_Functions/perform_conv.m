function out = perform_conv(u,v,ind_0,option)




%------------Add necessary zero-padding to make vectors symmetric----------
if (ind_0 < 1+(length(u)-1)/2)


for i = 1: -(ind_0 - 1+(length(u)-1)/2)
    u = [0,u];
    v = [0,v];
end
else
for i = 1: (ind_0 - 1+(length(u)-1)/2)
    u = [u,0];
    v = [v,0];
end
end
%---------------------------------

reversed_v = [];
for i = length(v):-1:1
    reversed_v = [reversed_v,v(i)];
end

if (option == 1)
    out = zeros(1,length(u));
    for k = 1:
end