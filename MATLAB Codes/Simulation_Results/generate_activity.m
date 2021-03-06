function generate_activity(m,n_exc,n_inh,p,T,mode,params)

%=============================Random Stimulus==============================
if (mode == 1)
    
    %----------------------In-Loop Initializations-------------------------
    theta = params(1);
    q = params(2);
    tau = params(3);
    %----------------------------------------------------------------------
    
    %----------------------Create an Erdos Reny Random Graph---------------    
    G = erdos_reny_inhibitory(m,n_exc,n_inh,p);        
    %----------------------------------------------------------------------
                
    %------------------------Record Sample States--------------------------
    S = zeros(T,n);                 % recorded stimulus states         
    R = zeros(T,1);                 % recorder output state            
                
    t_last = 0;                     % The last time the output neuron fired
    for i = 1:T    
        s = max(rand(1,n) < q,0);   % stimulus applied to n neurons            
        S(i,:) = s';                                
        denom = tau.^(0:1:i-t_last-1);                        
        h = sum(G*(S(i:-1:t_last+1,:)')./denom);
            
        R(i) = 1*(h/n > theta);        
        if (R(i))
            t_last =i;
        end
    end    
    %----------------------------------------------------------------------
%==========================================================================


%==========================IZHIKEVICH MODEL================================
elseif (mode == 2)       
    n_exc=200;                 n_inh=10;
    re=rand(n_exc,1);          ri=rand(n_inh,1);
    a=[0.02*ones(n_exc,1);     0.02+0.08*ri];
    b=[0.2*ones(n_exc,1);      0.25-0.05*ri];
    c=[-65+15*re.^2;        -65*ones(n_inh,1)];
    d=[8-6*re.^2;           2*ones(n_inh,1)];
    S=[rand(n_exc+n_inh,n_exc)<p,  -(rand(n_exc+n_inh,n_inh)<p)];
    states_exc = zeros(n_exc,T);
    states_inh = zeros(n_inh,T);

    v=-65*ones(n_exc+n_inh,1);    % Initial values of v
    u=b.*v;                 % Initial values of u
    firings=[];             % spike timings

    for t=1:T            % simulation of 1000 ms
        I=[5*randn(n_exc,1);2*randn(n_inh,1)]; % thalamic input
        fired=find(v>=30);    % indices of spikes
        states_exc(:,t) = v(1:n_exc)>=30;
        states_inh(:,t) = v(n_exc+1:n_exc+n_inh)>=30;
  
        firings=[firings; t+0*fired,fired];
        v(fired)=c(fired);
        u(fired)=u(fired)+d(fired);
        I=I+sum(S(:,fired),2);
        v=v+0.5*(0.04*v.^2+5*v+140-u+I); % step 0.5 ms
        v=v+0.5*(0.04*v.^2+5*v+140-u+I); % for numerical
        u=u+a.*(b.*v-u);                 % stability
    end
    
    states = [states_exc;states_inh];
    ind = 1+floor(n_exc*rand);
    G = S(ind,1:end);    
    R = states(ind,:);
    S = states(1:ind-1,:);
    S = [S;states(ind+1:end,:)];
    S = S';


%==========================================================================    

else
    error('Invalid state generating mode')
end