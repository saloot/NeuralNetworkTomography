function G = read_graph(ensmeble_count,mode,params_in)

%==============================INITIALIZATION==============================
% ensmeble_count = 1;
% n_exc = 240;
% n_inh = 60;

%==========================================================================

%=========================READ RESULTS FROM FILE===========================

if (mode == 1)
    n_exc = params(1);
    n_inh = params(2);
    n = n_exc + n_inh;
%--------------Load the Excitatory to Excitatory Connections---------------
fid = fopen(['./neurons_results/graph/graph_E2E_',num2str(ensmeble_count),'.txt'],'r');
if (fid > -1)
    g_exc2exc = fscanf(fid, '%d');
    fclose(fid);
else
    error('Invalid input file')
end
%--------------------------------------------------------------------------

%--------------Load the Excitatory to Inhibitory Connections---------------
fid = fopen(['./neurons_results/graph/graph_E2I_',num2str(ensmeble_count),'.txt'],'r');
if (fid > -1)
    g_exc2inh = fscanf(fid, '%d');
    fclose(fid);
else
    error('Invalid input file')
end
%--------------------------------------------------------------------------

%--------------Load the Inhibitory to Inhibitory Connections---------------
fid = fopen(['./neurons_results/graph/graph_I2I_',num2str(ensmeble_count),'.txt'],'r');
if (fid > -1)
    g_inh2inh = fscanf(fid, '%d');
    fclose(fid);
else
    error('Invalid input file')
end
%--------------------------------------------------------------------------

%--------------Load the Inhibitory to Excitatory Connections---------------
fid = fopen(['./neurons_results/graph/graph_I2E_',num2str(ensmeble_count),'.txt'],'r');
if (fid > -1)
    g_inh2exc = fscanf(fid, '%d');
    fclose(fid);
else
    error('Invalid input file')
end
%--------------------------------------------------------------------------

%==========================================================================



%====================REORGANIZE THE CONNECTIVITY MATRICES==================

%-----------------The Excitatory to Excitatory Connections-----------------
G_exc2exc = zeros(n_exc,n_exc);
for i = 1:n_exc
    G_exc2exc(i,:) = g_exc2exc(1+(i-1)*n_exc:i*n_exc);
end
%--------------------------------------------------------------------------

%-----------------The Excitatory to Inhibitory Connections-----------------
G_exc2inh = zeros(n_exc,n_inh);
for i = 1:n_exc
    G_exc2inh(i,:) = g_exc2inh(1+(i-1)*n_inh:i*n_inh);
end
%--------------------------------------------------------------------------

%-----------------The Inhibitory to Inhibitory Connections-----------------
G_inh2inh = zeros(n_inh,n_inh);
for i = 1:n_inh
    G_inh2inh(i,:) = -g_inh2inh(1+(i-1)*n_inh:i*n_inh);
end
%--------------------------------------------------------------------------

%-----------------The Inhibitory to Excitatory Connections-----------------
G_inh2exc = zeros(n_inh,n_exc);
for i = 1:n_inh
    G_inh2exc(i,:) = -g_inh2exc(1+(i-1)*n_exc:i*n_exc);
end
%--------------------------------------------------------------------------

%==========================================================================

G = [G_exc2exc,G_exc2inh;G_inh2exc,G_inh2inh];
elseif (mode == 2)
%----------------Load the Excitatory to Others Connections-----------------
fid = fopen(['../Neurons_data/graph/We_',num2str(ensmeble_count),'.txt'],'r');
if (fid > -1)
    g_exc = fscanf(fid, '%f');
    fclose(fid);
else
    error('Invalid input file')
end
%--------------------------------------------------------------------------

%----------------Load the Inhibitory to Others Connections-----------------
fid = fopen(['../Neurons_data/graph/Wi_',num2str(ensmeble_count),'.txt'],'r');
if (fid > -1)
    g_inh = fscanf(fid, '%f');
    fclose(fid);
else
    error('Invalid input file')
end
%--------------------------------------------------------------------------

%-----------------The Excitatory to Excitatory Connections-----------------
G_exc = zeros(n_exc,n);
for i = 1:n_exc
    G_exc(i,:) = sign(g_exc(1+(i-1)*n:i*n));
end
%--------------------------------------------------------------------------

%-----------------The Inhibitory to Excitatory Connections-----------------
G_inh = zeros(n_inh,n);
for i = 1:n_inh
    G_inh(i,:) = sign(g_inh(1+(i-1)*n:i*n));
end
%--------------------------------------------------------------------------

G = [G_exc;G_inh];

%=============================FEED FORWARD NETWORK=========================
elseif (mode == 3)
    params = params_in{1};
    n_f = params(1);
    n_o = params(2);
    n_inp = params(3);
    p1 = params(4);    
    delay_1 = params(5);
    f = params(6);
    
    file_path = params_in{2};
    file_name_ending = ['n_f_',num2str(n_f),'_n_o_',num2str(n_o),'_n_inp_',num2str(n_inp),'_p1_',num2str(p1),'_f_',num2str(f),'_d_',num2str(delay_1),'_',num2str(ensmeble_count)];
    
    %--------------Load the Excitatory to Others Connections---------------
    [file_path,'/Graphs/Wf_',file_name_ending,'.txt']
    fid = fopen([file_path,'/Graphs/Wf_',file_name_ending,'.txt'],'r');
    if (fid > -1)
        g = fscanf(fid, '%f');
        fclose(fid);
    else
        error('Invalid input file')
    end
    %----------------------------------------------------------------------  

    %---------------The Excitatory to Excitatory Connections---------------
    G = zeros(n_f,n_o);
    for i = 1:n_f
        G(i,:) = sign(g(1+(i-1)*n_o:i*n_o));
    end
    %----------------------------------------------------------------------

end
