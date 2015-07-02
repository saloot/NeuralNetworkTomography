function T = Infer_GLM(no_layers,n_exc,n_inh,PP,QQ,we_know_location)

%===================================INITIALIZATIONS======================================
global RefreshRate;  % Stimulus refresh rate (Stim frames per second)
RefreshRate = 100;

mkdir('../../Results/GLM/')

n_f = n_exc+n_inh;
if no_layers>1
    n_o = n_f(2);
end
n_f = n_f(1);
ensemble_count = 0; %1

nkt = 20;    % Number of time bins in filter;
DTsim = .1; % Bin size for simulating model & computing likelihood (in units of stimulus frames)

if no_layers > 1
    n_neurons = [n_f,n_o];
else
    n_neurons = n_f;
end

ttk = [-nkt+1:0]'; 
eval_str = '';

generate_data = 'R';                        % If 'S', the "stimulate-observe-rest" approach is followed. If equal to 'R', we are in the 'observe' mode 

%~~~~~~~~~~~~~~~Determine the Number of Pre-synaptic Neurons~~~~~~~~~~~~~~~
if we_know_location == 'F'
    n_presyn = n_f;
else
    n_presyn = sum(n_neurons);
    n_post = sum(n_neurons);
end
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


%~~~~~~~~~~~~~~~~~Initialize the Neural Structure~~~~~~~~~~~~~~~~~~~~~~~~~~
for i = 1:n_presyn
    eval(['ggsim',num2str(i),' = makeSimStruct_GLM(nkt,DTsim);']);  % Create GLM struct with default params
    eval_str = [eval_str,'ggsim',num2str(i),','];
end
eval_str_base = eval_str;
eval_str = eval_str(1:length(eval_str)-1);
eval(['ggsim = makeSimStruct_GLMcpl(',eval_str,');']);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%========================================================================================

%=========================IMPORTING DATA INTO A DICTIONARY===============================
if no_layers > 1
    file_name_ending = [num2str(n_exc(1)),'_',num2str(n_exc(2)),...
    '_n_inh_',num2str(n_inh(1)),'_',num2str(n_inh(2)),...
    '_p_0.0_0.299_0.0_R_1_d_0.0_9.0_0.0_',num2str(ensemble_count),'_q_0.4_G_',generate_data,'_l_'];

    file_name_ending = [num2str(n_exc(1)),'_',num2str(n_exc(2)),...
    '_n_inh_',num2str(n_inh(1)),'_',num2str(n_inh(2)),...
    '_p_0.0_',num2str(PP),'_0.0_R_1_d_0.0_9.0_0.0_',num2str(ensemble_count),'_q_',num2str(QQ),'_G_',generate_data,'_l_'];
else
   file_name_ending = [num2str(n_exc(1)),'_n_inh_',num2str(n_inh(1)),...
    '_p_',num2str(PP),'_R_1_d_10_',num2str(ensemble_count),'_q_',num2str(QQ),'_G_',generate_data,'_l_']; 
end
%file_name_ending = '50_40_n_inh_10_8_p_0.0_0.299_0.0_R_1_d_0.0_0.9_0.0_1_q_0.4_G_F_l_';
[S_times,T] = read_neural_spikes(file_name_ending,no_layers,n_neurons);
%========================================================================================

if we_know_location == 'F'
    inds = [1:n_f];
    neuron_range = [n_f+1:n_f+n_o];
else
    neuron_range = [1:n_post];
end

%========================CONSTRUCT THE RANGE OF SAMPLES NUMBERS==========================
tsp = S_times;
Stim = zeros(T,1);
t_step = floor((T-100)/5);
T_range = [100:t_step:T];
%T_range = [100, 1100, 2100, 3100, 4100, 5100];

for TT = T_range
    for i = neuron_range
        if (tsp{i}) & ( size(tsp{i},1)>1)
            sp_times = (tsp{i}<=TT).*tsp{i};
            indd = find(sp_times);
            if indd
                indd = indd(end);
                sp_times = sp_times(1:indd);
                tsp{i} = sp_times;
            else
                tsp{i} = [];
            end
        end
    end
    eval(['tsp_',num2str(TT),' = tsp;']);
end
%========================================================================================


%=============================PERFORM THE GLM ALGORITHM==================================

%---------------------------------The First Neuron---------------------------------------
if 0
i = 1;
sta0 = simpleSTC(Stim,tsp{i},nkt); % Compute STA 1
eval(['sta',num2str(i),' = reshape(sta0,nkt,[]);']);
inds = [1:i-1,i+1:n_o+n_f];
inds = [1:n_f];
    
eval(['gg0 = makeFittingStruct_GLM(sta',num2str(i),',DTsim,ggsim,1);']);  % Initialize params for fitting struct w/ sta
gg0.ih = [];%gg0.ih*0;  % Initialize to zero
gg0.dc = gg0.dc*0;  % Initialize to zero

gg0.tsp = tsp{i};     % cell 2 spike times (vector)
gg0.tsp2 = tsp(inds);  % spike trains from "coupled" cells (cell array of vectors)gg0.tspi = 1; % 1st spike to use for computing likelihood (eg, can ignore 1st n spikes)
gg0.tspi = 1; % 1st spike to use for computing likelihood (eg, can ignore 1st n spikes)

% Do ML estimation of model params
fprintf('Fitting first neuron\n');
opts = {'display', 'iter', 'maxiter', 250};
eval(['[gg',num2str(i),', negloglival',num2str(i),'] = MLfit_GLM(gg0,Stim,opts);']); % do ML (requires optimization toolbox)
%----------------------------------------------------------------------------------------
end 
  


base_str = 'n_';
for ll = 1:length(n_exc)
    base_str = [base_str,num2str(n_exc(ll)),'_'];
end
for ll = 1:length(n_inh)
    base_str = [base_str,num2str(n_inh(ll)),'_'];
end
base_str = [base_str,'P_',num2str(ceil(PP*100)),'_'];
base_str = [base_str,'Q_',num2str(ceil(QQ*100)),'_'];

for il = 1:length(neuron_range)
    i = neuron_range(il);
    T_itr = 0;
    for TT = T_range
        T_itr = T_itr + 1;
        opts = {'display', 'iter', 'maxiter', 500+50*T_itr};
        eval(['tsp = tsp_',num2str(TT),';']);
        Stim = zeros(TT,1);
        if (tsp{i}) & ( size(tsp{i},1)>1)
            %~~~~~~~~~~~~~~~~~~~~~~~~~Initialize the Weights~~~~~~~~~~~~~~~~~~~~~~~
            eval(['ggsim',num2str(i),' = makeSimStruct_GLM(nkt,DTsim);']);  % Create GLM struct with default params
            if we_know_location == 'F'
                eval_str = [eval_str_base,'ggsim',num2str(i),','];
                eval_str = eval_str(1:length(eval_str)-1);
            end
            eval(['ggsim = makeSimStruct_GLMcpl(',eval_str,');']);
            %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
            display(['--------------Neuron: ',num2str(i),'and T = ',num2str(TT),'------------------'])
            sta0 = simpleSTC(Stim,tsp{i},nkt); % Compute STA 1
            eval(['sta',num2str(i),' = reshape(sta0,nkt,[]);']);
            if we_know_location ~= 'F'
                inds = [1:i-1,i+1:n_post];
            end
    
            sta0 = simpleSTC(Stim,tsp{i},nkt); % Compute STA 1
            eval(['sta',num2str(i),' = reshape(sta0,nkt,[]);']);
    
    
            eval(['gg0 = makeFittingStruct_GLM(sta',num2str(i),',DTsim,ggsim,1);']);  % Initialize params for fitting struct w/ sta
            gg0.ih = gg0.ih*0;  % Initialize to zero
            gg0.dc = gg0.dc*0;  % Initialize to zero

            gg0.tsp = tsp{i};     % cell 2 spike times (vector)
            gg0.tsp2 = tsp(inds);  % spike trains from "coupled" cells (cell array of vectors)gg0.tspi = 1; % 1st spike to use for computing likelihood (eg, can ignore 1st n spikes)
            gg0.tspi = 1; % 1st spike to use for computing likelihood (eg, can ignore 1st n spikes)


    
            eval(['gg0',num2str(i),' = gg0;']);             % initial parameters for fitting 
            eval(['gg0',num2str(i),'.tsp =tsp{i};']);       % cell i spike times (vector)
            eval(['gg0',num2str(i),'.tsp2 = tsp(inds);']);  % spike trains from "coupled" cells (cell array of vectors)
    
            eval(['gg0',num2str(i),'.kt = inv(gg0.ktbas''*gg0.ktbas)*gg0.ktbas''*sta',num2str(i),';']); % Project STA2 into basis 
            eval(['gg0',num2str(i),'.k = gg0',num2str(i),'.ktbas*gg0',num2str(i),'.kt;']); % Project STA onto basis for fitting
    
            eval(['[gg',num2str(i),', negloglival',num2str(i),'] = MLfit_GLM(gg0',num2str(i),',Stim,opts);']);
        else
            eval(['gg',num2str(i),'=[];']);
        end
        eval(['save([''../../Results/GLM/GLM_'',base_str,''N_'',num2str(i),''_T_'',num2str(TT),''_'',num2str(we_know_location)],[''gg'',num2str(i)])']);        
    end
    %eval(['clear gg',num2str(i)]);
end
%========================================================================================

%========================COMPARE THE RESULTS TO THE ACTUAL GRAPH=========================

%========================================================================================
