function compare_results_GLM(no_layers,n_exc,n_inh,PP,QQ,we_know_location,T)

%------------------------------Initializations-----------------------------


generate_data = 'R';                       % If 'S', the "stimulate-observe-rest" approach is followed. If equal to 'R', we are in the 'observe' mode 

if no_layers == 2
    n_f = n_exc+n_inh;
    n_o = n_f(2);
    n_f = n_f(1);
    n_neurons = [n_f,n_o];
else
    network_structure = 'R';
    n_neurons = [n_f];
    n_f = n_exc+n_inh;    
end

ensemble_count = 0; %1


N = sum(n_exc)+sum(n_inh);

n_layer_1 = n_exc(1) + n_inh(1);
if no_layers > 1
    n_layer_2 = n_exc(2) + n_inh(2);
end



T_range = [100,2779,5458,8137,10816,13495]; 
T_range = [100,2580,5060,7540,10020,12500]; 
T_range = [100, 1330, 2560, 3790, 5020, 6250];
T_range = [100, 1000, 2000, 3000, 4000, 5000];
T_range = [100, 1100, 2100, 3100, 4100, 5100];
T_range = [100:t_step:T];

B_e = zeros(1,length(T_range));
B_i = zeros(1,length(T_range));
B_v = zeros(1,length(T_range));
S_e = zeros(1,length(T_range));
S_i = zeros(1,length(T_range));
S_v = zeros(1,length(T_range));
%--------------------------------------------------------------------------

%--------------------------Read Weight Matrix From File--------------------
folder_base = '../../Data/Graphs/';
if no_layers == 2
    file_name_base = ['L_',num2str(no_layers),'_n_exc_',num2str(n_exc(1)),'_',num2str(n_exc(2)),...
    '_n_inh_',num2str(n_inh(1)),'_',num2str(n_inh(2)),...
    '_p_0.0_',num2str(PP),'_0.0_R_1_d_0.0_9.0_0.0_',num2str(ensemble_count),'_l_'];
else
    file_name_base = ['L_',num2str(no_layers),'_n_exc_',num2str(n_exc(1)),'_n_inh_',num2str(n_inh(1)),...
        '_p_',num2str(PP),'_R_1_d_10_',num2str(ensemble_count),'_l_'];
end

%~~~~~~~~~~~~~~~Read the Weights for Layer 0 to Layer 0~~~~~~~~~~~~~~~~~~~~
W1 = importdata([folder_base,'We_',file_name_base,'0_to_0.txt']);
W1i = importdata([folder_base,'Wi_',file_name_base,'0_to_0.txt']);
W_temp = [W1;W1i];
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%~~~~~~~~~~~~~~~Read the Weights for Layer 0 to Layer 1~~~~~~~~~~~~~~~~~~~~
if no_layers == 2
    W1 = importdata([folder_base,'We_',file_name_base,'0_to_1.txt']);
    W1i = importdata([folder_base,'Wi_',file_name_base,'0_to_1.txt']);
    W_temp2 = [W1;W1i];
    W_temp = [W_temp,W_temp2];
    
    %~~~~~~~~~~~~~~~Read the Weights for Layer 1 to Layer 1~~~~~~~~~~~~~~~~~~~~
    W1 = importdata([folder_base,'We_',file_name_base,'1_to_1.txt']);
    W1i = importdata([folder_base,'Wi_',file_name_base,'1_to_1.txt']);
    W_temp2 = [W1;W1i];
    W_temp = [W_temp;zeros(n_exc(2)+n_inh(2),n_exc(1)+n_inh(1)),W_temp2];
end

%~~~~~~~~~~~~~~~Determine the Number of Pre-synaptic Neurons~~~~~~~~~~~~~~~
if we_know_location == 'Y'
    n_presyn = n_layer_1;
    neuron_range = [n_layer_1+1:n_layer_1+n_layer_2];
    estimated_weights = zeros(n_layer_1,n_layer_2);
    estimated_weights_signed = zeros(n_layer_1,n_layer_2);
else
    neuron_range = [1:N];
    n_presyn = N;
    estimated_weights = zeros(N,N);
    estimated_weights_signed = zeros(N,N);
end
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%--------------------------------------------------------------------------


%-----------------------Read Weight for the GLM Method---------------------
base_str = 'n_';
for ll = 1:length(n_exc)
    base_str = [base_str,num2str(n_exc(ll)),'_'];
end
for ll = 1:length(n_inh)
    base_str = [base_str,num2str(n_inh(ll)),'_'];
end
base_str = [base_str,'P_',num2str(ceil(100*PP)),'_'];
base_str = [base_str,'Q_',num2str(ceil(100*QQ)),'_'];

itr_T = 0;
for TT = T_range
    itr_T = itr_T + 1;
    for il = 1:length(neuron_range)
        
        neuron_ind = neuron_range(il);
        if we_know_location == 'Y'
            eval(['load([''../../Results/GLM/GLM_'',base_str,''N_'',num2str(neuron_ind),''_T_'',num2str(TT),''_F''])']);
        else
            eval(['load([''../../Results/GLM/GLM_'',base_str,''N_'',num2str(neuron_ind),''_T_'',num2str(TT),''_R''])']);
        end
        eval(['incoming_filters = gg',num2str(neuron_ind),';']);
    
        try (incoming_filters.ih2);
            ee = 1;
        catch
            ee = 0;
        end
        if ee
            incoming_filters = incoming_filters.ih2;
            eval(['self_filters = gg',num2str(neuron_ind),'.ih;']);
            %incoming_filters = [incoming_filters(:,1:neuron_ind-1),self_filters,incoming_filters(:,neuron_ind:end)];
            if we_know_location ~= 'Y'
                incoming_filters = [incoming_filters(:,1:il-1),self_filters,incoming_filters(:,il:end)];
            end
            eval(['temp_or = (gg',num2str(neuron_ind),'.ihbas*(incoming_filters));']);
            temp2 = sum(abs(temp_or));
            temp = sum(temp_or); 
    
            %temp = temp - mean(temp);
            %temp = temp/var(temp);
            %temp2 = temp2 - mean(temp2);
            %temp2 = temp2/var(temp2);
            if we_know_location == 'Y'
                temp = temp(1:end);
                temp2 = temp2(1:end);
            end
    
            estimated_weights(:,il) = temp2';%temp2';
            estimated_weights_signed(:,il) = temp';
            
            estimated_weights(:,il) = max(incoming_filters);
            [vals,ind] = max(abs(temp_or));
            estimated_weights_signed(:,il) = vals.*(max(temp_or)== vals) + min(temp_or).*(abs(min(temp_or))== vals);
            %estimated_weights_signed(:,il) = sum(incoming_filters>0);
        end
    end
    %--------------------------------------------------------------------------

    %---------------------------Calculate Beliefs------------------------------
    W_tot = W_temp;
    Beliefs_vals = estimated_weights.* estimated_weights_signed;
    Beliefs_vals = estimated_weights_signed;%./(abs(estimated_weights_signed)+0.001);
    %Beliefs_vals = Beliefs_vals/max(max(abs(Beliefs_vals)));
    if network_structure == 'R'
        W_e = (W_tot>0);
        B_e(1,itr_T) = mean((sum(Beliefs_vals .* W_e))./(0.0001+sum(W_e)));
        S_e(1,itr_T) = std((sum(Beliefs_vals .* W_e))./(0.0001+sum(W_e)));

        W_i = (W_tot<0);
        B_i(1,itr_T) = mean((sum(Beliefs_vals .* W_i))./(0.0001+sum(W_i)));
        S_i(1,itr_T) = std((sum(Beliefs_vals .* W_i))./(0.0001+sum(W_i)));

        W_v = (W_tot==0);
        B_v(1,itr_T) = mean((sum(Beliefs_vals .* W_v))./(sum(0.0001+W_v))); 
        S_v(1,itr_T) = std((sum(Beliefs_vals .* W_v))./(sum(0.0001+W_v))); 
    else
        if we_know_location == 'Y'        
            W_e = (W_tot(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2)>0);
            B_e(1,itr_T) = mean((sum(Beliefs_vals .* W_e))./(0.0001+sum(W_e)));
            S_e(1,itr_T) = std((sum(Beliefs_vals .* W_e))./(0.0001+sum(W_e)));

            W_i = (W_tot(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2)<0);
            B_i(1,itr_T) = mean((sum(Beliefs_vals .* W_i))./(0.0001+sum(W_i)));
            S_i(1,itr_T) = std((sum(Beliefs_vals .* W_i))./(0.0001+sum(W_i)));

            W_v = (W_tot(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2)==0);
            B_v(1,itr_T) = mean((sum(Beliefs_vals .* W_v))./(sum(0.0001+W_v))); 
            S_v(1,itr_T) = std((sum(Beliefs_vals .* W_v))./(sum(0.0001+W_v)));   
        else
            W_e = (W_tot(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2)>0);
            B_e(1,itr_T) = mean((sum(Beliefs_vals(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2) .* W_e))./(0.0001+sum(W_e)));
            S_e(1,itr_T) = std((sum(Beliefs_vals(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2) .* W_e))./(0.0001+sum(W_e)));
      
            W_i = (W_tot(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2)<0);
            B_i(1,itr_T) = mean((sum(Beliefs_vals(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2) .* W_i))./(0.0001+sum(W_i)));
            S_i(1,itr_T) = std((sum(Beliefs_vals(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2) .* W_i))./(0.0001+sum(W_i)));
        
            W_v = (W_tot(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2)==0);
            B_v(1,itr_T) = mean((sum(Beliefs_vals(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2) .* W_v))./(0.0001+sum(W_v)));
            S_v(1,itr_T) = std((sum(Beliefs_vals(1:n_layer_1,n_layer_1+1:n_layer_1+n_layer_2) .* W_v))./(sum(0.0001+W_v))); 
        
            W_v_r = (W_tot(n_layer_1+1:n_layer_1+n_layer_2,n_layer_1+1:n_layer_1+n_layer_2)==0);
            B_v_r(1,itr_T) = mean((sum(Beliefs_vals(n_layer_1+1:n_layer_1+n_layer_2,n_layer_1+1:n_layer_1+n_layer_2) .* W_v_r))./(0.0001+sum(W_v_r)));
            S_v_r(1,itr_T) = std((sum(Beliefs_vals(n_layer_1+1:n_layer_1+n_layer_2,n_layer_1+1:n_layer_1+n_layer_2) .* W_v_r))./(sum(0.0001+W_v_r))); 
        
            %plot(B_e,'r');hold on;plot(B_i,'b');plot(B_v,'g');plot(B_v_r,'g--');
            %title('Connections Originating coming into Post-synaptic Neurons');

            W_v_b = (W_tot(n_layer_1+1:n_layer_1+n_layer_2,1:n_layer_1)==0);
            B_v_b(1,itr_T) = mean((sum(Beliefs_vals(n_layer_1+1:n_layer_1+n_layer_2,1:n_layer_1) .* W_v_b))./(0.0001+sum(W_v_b)));
            S_v_b(1,itr_T) = std((sum(Beliefs_vals(n_layer_1+1:n_layer_1+n_layer_2,1:n_layer_1) .* W_v_b))./(sum(0.0001+W_v_b))); 
        
            W_v_p = (W_tot(1:n_layer_1,1:n_layer_1)==0);
            B_v_p(1,itr_T) = mean((sum(Beliefs_vals(1:n_layer_1,1:n_layer_1) .* W_v_p))./(0.0001+sum(W_v_b)));
            S_v_p(1,itr_T) = std((sum(Beliefs_vals(1:n_layer_1,1:n_layer_1) .* W_v_p))./(sum(0.0001+W_v_p))); 
            %figure;plot(B_v_b,'g');hold on;plot(B_v_r,'g--');
            %title('Connections Originating coming into Pre-synaptic Neurons');
        end
    end
    
    %------------------------Save the Belief Matrix------------------------
    if we_know_location == 'Y'
        file_name = ['../../Results/Inferred_Graphs/W_',file_name_base,'0_to_1_I_8_Loc_',num2str(we_know_location),'_Pre_A_G_',...
        num2str(generate_data),'_X_1_Q_',num2str(QQ),'_T_',num2str(TT),'.txt'];
    else
        file_name = ['../../Results/Inferred_Graphs/W_',file_name_base(1:end-2),'I_8_Loc_',num2str(we_know_location),'_Pre_A_G_',...
        num2str(generate_data),'_X_1_Q_',num2str(QQ),'_T_',num2str(TT),'.txt'];
    end
    
    fid = fopen(file_name,'w');
    [m,n] = size(Beliefs_vals);
    for ii = 1:m
        fprintf(fid,'%f\t',Beliefs_vals(ii,:));
        fprintf(fid,'\n');  
    end
    fclose(fid);
    %dlmwrite(file_name,Beliefs_vals,'delimiter','\t','newline','unix','precision',6);
    %----------------------------------------------------------------------

end
%--------------------------------------------------------------------------


%---------------------------Compare the Weights----------------------------
errorbar(B_e,S_e,'r');hold on;errorbar(B_i,S_i,'b');errorbar(B_v,S_v,'g');
title('Connections Originating coming into Post-synaptic Neurons');
    
figure;
plot(S_e,'r');hold on;plot(S_i,'b');plot(S_v,'g');
title('Variance of Connections Originating coming into Post-synaptic Neurons');

%==============================SAVE THE RESULTS============================

%==========================================================================


W_tot = (W_temp >0) + 0.5 * (W_temp<0);
subplot(131);
imshow(W_tot);

W_estim = estimated_weights_signed-min(min(estimated_weights_signed));
W_estim = estimated_weights_signed/max(max(W_estim));

subplot(132);
imshow((W_estim));

W_estim_s = estimated_weights_signed-min(min(estimated_weights_signed));
W_estim_s = estimated_weights_signed/max(max(W_estim_s));

W_estim = (estimated_weights>0.75).*(estimated_weights_signed);

subplot(133);
W_tem = imadjust(W_estim);
imshow(imadjust(W_tem));

W_bin = (W_tem > 0.75) - (W_tem < 0.25).*(W_tem>0);
%--------------------------------------------------------------------------


