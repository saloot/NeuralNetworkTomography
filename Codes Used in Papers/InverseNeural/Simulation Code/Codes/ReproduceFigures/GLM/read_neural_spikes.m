function [S_times,t_max] = read_neural_spikes(file_name_ending,no_layers,n_neurons)

t_max = 0;

%------------------------Read the Spikes from File-------------------------
for l = 1:no_layers
    fid = fopen(['../../Data/Spikes/S_times_L_',num2str(no_layers),'_n_exc_',file_name_ending,num2str(l-1),'.txt'],'r');
    if (fid > -1)
        eval(['s_l',num2str(l-1),' = fscanf(fid, ''%f'',[2,inf]);']);        
        fclose(fid);
    else
        error('Invalid input file')
    end
end    
%--------------------------------------------------------------------------

%-------------------------Initialize the Spike Matrices--------------------
neuron_count_base = 1;
S_times = {};
for i = 1:sum(n_neurons)
   S_times{i} = [];
end
t_base = 0;
%--------------------------------------------------------------------------

%------------------------------Load the Data-------------------------------
for ll = 1:no_layers
    eval(['spikes = s_l',num2str(ll-1),';']);
    [~,l] = size(spikes);    
    for i = 1:l
        if (spikes(1,i) == -2) 
            t_base = t_base + 1;
                %if t_base > 2 * 500
                %    break;
                %end
        else    
            t = t_base + round(1000*spikes(2,i));
            if t>t_max
                t_max = t;
            end
            neuron_count = neuron_count_base+spikes(1,i);
            
            s_times = S_times{neuron_count};
            S_times{neuron_count} = [s_times;t];            
        end    
    end
    neuron_count_base = neuron_count_base + n_neurons(ll);
end
%--------------------------------------------------------------------------

t_max = ceil(t_max);
