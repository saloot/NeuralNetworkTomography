
%===========================INITIALIZATION=================================
no_averaging_itrs = 30;         % number of times we perform each simulation for the sake of averging

a=clock;                                % Initialize the seed for random number generation with the clock value.
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*a))); 

parameter_range = [100:500:2100];
% parameter_range = [0.01:.05:.75];
% parameter_range = [100:200:800];
% parameter_range = [0.01:.02:.25];
success_measure = zeros(2,length(parameter_range));
acc_theory = zeros(1,length(parameter_range));
err_theory = zeros(1,length(parameter_range));

%==========================================================================


%=======================LOOP OVER THE PARAMETER============================
for itr = 1:length(parameter_range)

    %---------------------In-loop Initializations--------------------------
    theta = 0.05;   % update threshold
   
    p = 0.3;                        % connection probability
    n = 200;                        % number of neurons
    T = parameter_range(itr);                       % number of sample recordings

    acc_ave = 0;                    % average number of correctly infered edges
    err_ave = 0;                    % average number of mistakenly infered edges + missed edges

    q = theta/p;                    % stimulus firing probability
    %----------------------------------------------------------------------

    %--------------Average over an Ensemble of Random Graphs---------------
    for av_itr = 1:no_averaging_itrs
    
        %--------------------Create an Erdos Reny Random Graph-------------
        G = erdos_reny(1,n,p);
        %------------------------------------------------------------------

        
        %----------------------Record Sample States------------------------
        R = zeros(T,n+1);               % recorded states 
        for i = 1:T    
            s = max(rand(1,n) < q,0);   % stimulus applied to n neurons
            R(i,1:n) = s';
            R(i,n+1) = (G*s'/n > theta);
        end
        %------------------------------------------------------------------
        
        %------------------Infer the Connectivity Matrix-------------------
        W = zeros(1,n);
        % R = (R==0).*(-1*ones(T,n+1)) + R;
        for i=1:n
            W(i) = R(:,n+1)'*R(:,i)/T;
        end
        W2 = W;
        %------------------------------------------------------------------


        %------------------Determine the Weight Threshold------------------
        [W_sorted,ind] = sort(W);
        w_thr = determine_weight_thr(n,p,q,theta);
        W = (W>w_thr);
        W = zeros(1,n);
        W(ind(end-sum(G):end)) = (W_sorted(end-sum(G):end)>0);
        %------------------------------------------------------------------

        
        %--------------------Calculate the Accuracy-------------------------
        acc = sum(W.*G);
        err = abs(sum(sign(W+G)-W.*G));
        acc_ave = acc_ave + acc/sum(G);
        err_ave = err_ave + err/(n-sum(G));
        %------------------------------------------------------------------

    end
    
    %-----------Calculate Theoretical Bounds and Store Results-------------
    success_measure(1,itr) = acc_ave/av_itr;
    success_measure(2,itr) = err_ave/av_itr;
    [p1_T,p2_T] = simple_hebb_rule_theory(n,p,q,T,theta,0);
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