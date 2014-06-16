
%===========================INITIALIZATION=================================
no_averaging_itrs = 60;         % number of times we perform each simulation for the sake of averging

a=clock;                                % Initialize the seed for random number generation with the clock value.
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*a))); 

% parameter_range = [100:500:4100];
parameter_range = [.2:.05:.75];
% parameter_range = [100:200:800];
% parameter_range = [0.01:.02:.25];
success_measure = zeros(2,length(parameter_range));
acc_theory = zeros(1,length(parameter_range));
err_theory = zeros(1,length(parameter_range));
hist_window = 2;                    % the window considered to calculate the history
%==========================================================================


%=======================LOOP OVER THE PARAMETER============================
for itr = 1:length(parameter_range)

    %---------------------In-loop Initializations--------------------------
    theta = .01;                     % update threshold
   
    p = parameter_range(itr);       % connection probability
    n = 200;                        % number of neurons
    T = round(3000/(1-p));                       % number of sample recordings
  
    acc_ave = 0;                    % average number of correctly infered edges
    err_ave = 0;                    % average number of mistakenly infered edges + missed edges

    q = theta/p;                    % stimulus firing probability
    %----------------------------------------------------------------------

    %--------------Average over an Ensemble of Random Graphs---------------
    for av_itr = 1:no_averaging_itrs
    
        %--------------------Create an Erdos Reny Random Graph-------------
        G = erdos_reny(n,n,p);
        %------------------------------------------------------------------

        
        %----------------------Record Sample States------------------------
        S = zeros(T,n);                 % recorded stimulus
        R = zeros(T,n);                 % recorded responses
        for i = 1:T    
            s = max(rand(1,n) < q,0);   % stimulus applied to n neurons
            S(i,:) = s';
            R(i,:) = (s*G/n > theta).*(sum(R(max(1,i-hist_window):i-1,:))==0);
        end
        %------------------------------------------------------------------
        
        %------------------Infer the Connectivity Matrix-------------------
        W = zeros(n,n);
        % R = (R==0).*(-1*ones(T,n+1)) + R;
        for j = 1:n
            for i=1:n
                W(j,i) = S(:,j)'*R(:,i)/T;
            end
        end
        W2 = W;
        %------------------------------------------------------------------


        %------------------Determine the Weight Threshold------------------
        
%         w_thr = determine_weight_thr(n,p,q,theta);
%         W = (W>w_thr);
        W = zeros(n,n);
        for i = 1:n
            [W_sorted,ind] = sort(W2(:,i));
            W(ind(end-round(p*(n-1)):end),i) = (W_sorted(end-round(p*(n-1)):end)>0);
            W(i,i) = 0;
        end
        %------------------------------------------------------------------

        
        %--------------------Calculate the Accuracy-------------------------
        acc = sum(sum(W.*G));
        err = sum(sum(abs(sign(W+G)-W.*G)));
        acc_ave = acc_ave + acc/sum(sum(G));
        err_ave = err_ave + err/(n*(n-1)-sum(sum(G)));
        %------------------------------------------------------------------
        111;
    end
    
    %-----------Calculate Theoretical Bounds and Store Results-------------
    success_measure(1,itr) = acc_ave/av_itr;
    success_measure(2,itr) = err_ave/av_itr;
    [p1_T,p2_T] = simple_hebb_rule_theory(n,p,q,T,theta,hist_window);
    acc_theory(itr) =p1_T;
     err_theory(itr) = p2_T;
    %----------------------------------------------------------------------    
end
%==========================================================================

%==============================PLOT RESULTS================================
figure
plot(parameter_range,success_measure(1,:),'r-*')
hold on
plot(parameter_range,success_measure(2,:),'k-*')
plot(parameter_range,acc_theory,'r')
plot(parameter_range,err_theory,'k')
title(['q=',num2str(q),' p=',num2str(p),' theta=',num2str(theta),' n=',num2str(n)])
legend('No. correct edges','No. false edges')
xlabel('T')
%==========================================================================