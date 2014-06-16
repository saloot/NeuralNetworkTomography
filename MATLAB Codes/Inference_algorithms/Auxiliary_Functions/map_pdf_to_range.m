function pdf_out = map_pdf_to_range(pdf_base,range_base,range_tot)

pdf_out = zeros(1,length(range_tot));

for i = 1:length(pdf_base)    
    p = pdf_base(i);
    r = range_base(i);
    e = abs(range_tot - r);
    [val,ind] = min(e);
    pdf_out(ind) = p;            
end