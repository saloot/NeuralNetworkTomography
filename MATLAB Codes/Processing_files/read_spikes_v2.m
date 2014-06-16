function S = read_spikes_v2(ensmeble_count,params_in,mode)

%==============================INITIALIZATION==============================
% ensmeble_count = 1;
% n_exc = 240;
% n_inh = 60;
%==========================================================================

%=========================READ RESULTS FROM FILE===========================


if (mode == 1)
    n = n_exc + n_inh;
    %---------------------Load the Excitatory Spikes-----------------------
    fid = fopen(['../Neurons_data/spikes/spikes_E_',num2str(ensmeble_count),'.txt'],'r');
    if (fid > -1)
        s_exc = fscanf(fid, '%f %f',[2,inf]);
        s_exc = s_exc';
        fclose(fid);
    else
        error('Invalid input file')
    end
    %----------------------------------------------------------------------

    %---------------------Load the Inhibitory Spikes-----------------------
    fid = fopen(['../Neurons_data/spikes/spikes_I_',num2str(ensmeble_count),'.txt'],'r');
    if (fid > -1)
        s_inh = fscanf(fid, '%f %f',[2,inf]);
        s_inh = s_inh';
        fclose(fid);
    else
        error('Invalid input file')
    end
    %----------------------------------------------------------------------
elseif (mode == 2)
    %---------------------Load the Excitatory Spikes-----------------------
    fid = fopen(['../Neurons_data/spikes/S_times_',num2str(ensmeble_count),'.txt'],'r');
    if (fid > -1)
        s = fscanf(fid, '%f',[1,inf]);        
        fclose(fid);
    else
        error('Invalid input file')
    end
    %----------------------------------------------------------------------

elseif (mode == 3)
    %--------------Load the Spikes of the Feed-Forward Networks------------
    params = params_in{1}
    n_f = params(1);
    n_o = params(2);
    n_inp = params(3);
    p1 = params(4);
    delay_1 = params(5);
    T = params(6);
    f = params(7);    
    file_path = params_in{2};
    file_name_ending = ['n_f_',num2str(n_f),'_n_o_',num2str(n_o),'_n_inp_',num2str(n_inp),'_p1_',num2str(p1),'_f_',num2str(f),'_d_',num2str(delay_1),'_',num2str(ensmeble_count)];
    
    fid = fopen([file_path,'/Spikes/S_times_l1_',file_name_ending,'.txt'],'r');
    if (fid > -1)
        s_l1 = fscanf(fid, '%f',[1,inf]);        
        fclose(fid);
    else
        error('Invalid input file')
    end
    
    fid = fopen([file_path,'/Spikes/S_times_l2_',file_name_ending,'.txt'],'r');
    if (fid > -1)
        s_l2 = fscanf(fid, '%f',[1,inf]);        
        fclose(fid);
    else
        error('Invalid input file')
    end
    %----------------------------------------------------------------------    
else
    error('Invalid spike mode!')
end
%==========================================================================


%==========================REORGANIZE THE SPIKES===========================

if (mode == 1)
    %------------------------The Excitatory Spikes-------------------------
    S_exc = zeros(n_exc,T);
    [l,~] = size(s_exc);
    for i = 1:l
        if (s_exc(i,2) > 0)
            t = round(10000*s_exc(i,2));
            S_exc(s_exc(i,1)+1,t) = 1;
        end    
    end
    %----------------------------------------------------------------------

    %------------------------The Inhibitory Spikes-------------------------
    S_inh = zeros(n_inh,T);
    [l,~] = size(s_inh);
    for i = 1:l
        if (s_inh(i,2) > 0)
            t = round(10000*s_inh(i,2));
            S_inh(s_inh(i,1)+1,t) = 1;
        end    
    end
    %----------------------------------------------------------------------
    
    S = [S_exc;S_inh];
    
elseif (mode == 2)
    
    %------------------------------All Spikes------------------------------
    neuron_count = 1;
    S = cell(1,n);
    l = length(s);
    timings = [];
    for i = 1:l
        if (s(i) == -1)
            S{neuron_count} = timings;
            neuron_count = neuron_count + 1;
            timings = [];
        else
            timings = [timings,round(10000*s(i))];            
        end    
    end
    %----------------------------------------------------------------------
    
elseif (mode == 3)
    
    %--------------Load the Spikes of the Feed-Forward Networks------------    
    n_o = params(2);
        
    neuron_count = 1;
    S_l1 = cell(1,n_f);
    S_l2 = cell(1,n_o);
    
    l = length(s_l1);
    timings = [];
    for i = 1:l
        if (s_l1(i) == -1)
            S_l1{neuron_count} = timings;
            neuron_count = neuron_count + 1;
            timings = [];
        else
            timings = [timings,round(10000*s_l1(i))];            
        end    
    end
   
    neuron_count = 1;
    l = length(s_l2);
    timings = [];
    for i = 1:l
        if (s_l2(i) == -1)
            S_l2{neuron_count} = timings;
            neuron_count = neuron_count + 1;
            timings = [];
        else
            timings = [timings,round(10000*s_l2(i))];
        end    
    end 
    %----------------------------------------------------------------------
    S{1} = S_l1;
    S{2} = S_l2;
else
    error('Invalid spike mode!')
end

%==========================================================================
