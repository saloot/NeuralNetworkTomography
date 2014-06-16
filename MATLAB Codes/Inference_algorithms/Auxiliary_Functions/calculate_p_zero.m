function p_1_0 = calculate_p_zero(d_plus,d_minus,q,x_min,x_max,theta,tau,T_fire,delta)

range_resolution = 10^floor(log10(1/tau^T_fire));
range_tot = [x_min:range_resolution:x_max];
range_base = [x_min:x_max];
range = [];

pdf_out = zeros(1,length(range));
range_temp = [x_min:x_max];

pdf_plus = zeros(1,length(range_temp));
pdf_minus = zeros(1,length(range_temp));

d_range = d_plus;
for i = 0:d_plus 
   pdf_plus(abs(x_min) + 1+i) = nchoosek(d_range,i) * (q^i) * ((1-q)^(d_range-i));
end

d_range = d_minus;
for i = 0:d_minus
   pdf_minus(abs(x_min) + 1-i) = nchoosek(d_range,i) * (q^i) * ((1-q)^(d_range-i));
end


pdf_base = conv(pdf_plus,pdf_minus,'same');

pdf_out = [];
for k = 0:T_fire
    range_temp = range_base/(tau^k);
    range = [range,range_temp];
end
[range,ind] = sort(range);



pdf_out = map_pdf_to_range(pdf_base,range_base,range_tot);
    

for k = 1:T_fire
   pdf_temp = map_pdf_to_range(pdf_base,range_base/tau^k,range_tot);   
   temp = conv(pdf_temp,pdf_out,'same');
%    temp = ifft(fft(pdf_out).*fft(pdf_temp));
%    temp = ifftshift((fftshift(pdf_out).*fftshift(pdf_temp)));
%    temp = temp/sum(temp);
   pdf_out = temp;
end


thr = theta*(d_plus+d_minus);
p_1_0 = sum(pdf_out.*(range_tot>=thr));


 