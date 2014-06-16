function V = read_votltages(n_exc,n_inh,ensmeble_count,T)

%==============================INITIALIZATION==============================
% ensmeble_count = 1;
% n_exc = 240;
% n_inh = 60;
%==========================================================================

%=========================READ RESULTS FROM FILE===========================

%-----------------------Load the Excitatory Voltages-------------------------
fid = fopen(['./neurons_results/voltage/voltage_E_',num2str(ensmeble_count),'.txt'],'r');
if (fid > -1)
    V_exc = fscanf(fid, '%f %f',[T,inf]);
    V_exc = V_exc';
    fclose(fid);
    if (size(V_exc,1) ~= n_exc)
        error('Invalid format!')
    end
else
    error('Invalid input file')
end
%--------------------------------------------------------------------------

%-----------------------Load the Inhibitory Voltages-------------------------
fid = fopen(['./neurons_results/voltage/voltage_I_',num2str(ensmeble_count),'.txt'],'r');
if (fid > -1)
    V_inh = fscanf(fid, '%f %f',[T,inf]);
    V_inh = V_inh';    
    fclose(fid);
    if (size(V_inh,1) ~= n_inh)
        error('Invalid format!')
    end
else
    error('Invalid input file')
end
%--------------------------------------------------------------------------

%==========================================================================


V = [V_exc;V_inh];

