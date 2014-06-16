function S_times = detect_arriving_spikes(V,thr_exc,thr_inh)

%==============================INITIALIZATION==============================
% ensmeble_count = 1;
% n_exc = 240;
% n_inh = 60;
l = length(V);
V_diff = diff(V);
S_times = zeros(1,l);
spike_amp = .14;
Delta_diff_V = diff(V_diff);
%==========================================================================


%=====================CALCULATE THE VOLTAGE DIFFERENCE=====================
for i = 1:l-2    
    if (Delta_diff_V(i) > thr_exc)    
        if (Delta_diff_V(i) < 29)
            S_times(i) = round(Delta_diff_V(i)/spike_amp);
        end
    elseif (Delta_diff_V(i) < thr_inh)    
        if (Delta_diff_V(i) > -(29))
            S_times(i) = -ceil(-Delta_diff_V(i)/spike_amp);
        end
    end
end
%==========================================================================

