function [G,S,R] = generate_activity(m,n_exc,n_inh,p,T,mode,params,graph_mode)

a=clock;                                % Initialize the seed for random number generation with the clock value.
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*a))); 

n = n_exc+n_inh;


%=============================Random Stimulus==============================
if (mode == 1)
    
    %----------------------In-Loop Initializations-------------------------
    theta = params(1);
    q = params(2);
    tau = params(3);
    %----------------------------------------------------------------------
    
    %----------------------Create an Erdos Reny Random Graph---------------    
    if (graph_mode == 1)        
        G = erdos_reny_inhibitory(m,n_exc,n_inh,p);        
    else
        deg_min = params(4);
        deg_max = params(5);
        gamma = params(6);
                
        % G = power_law_inhibitory(m,n_exc,n_inh,deg_min,deg_max,gamma);
        
        d_plus = params(4);
        d_minus = params(5);
        G = [];        
        temp = zeros(1,n);
        ind = randperm(n_exc);
    
        temp(ind(1:d_plus)) = 1;
    
        ind = randperm(n_inh);
        if (d_minus > 0)
            temp(n_exc + ind(1:d_minus)) = -1;
        end     
        G = [G,temp];    
    end
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
        
        if (graph_mode == 1)
            R(i) = 1*(h/n > theta);        
        else
            R(i) = 1*(h/(sum(abs(G))) > theta);
        end
        if (R(i))
            t_last =i;
        end
    end    
    %----------------------------------------------------------------------
%==========================================================================


%==========================IZHIKEVICH MODEL================================
elseif (mode == 2)       
    
    if (params == 1)                    % Real weights        
        re=rand(n_exc,1);          ri=rand(n_inh,1);
%         a=[0.02*ones(n_exc,1);     0.02+0.08*ri];
%         b=[0.2*ones(n_exc,1);      0.25-0.05*ri];
%         c=[-65+15*re.^2;        -65*ones(n_inh,1)];
%         d=[8-6*re.^2;           2*ones(n_inh,1)];
        a=[0.02*ones(n_exc,1);     0.02+0.08*ri];
        b=[0.4*ones(n_exc,1);      0.25-0.05*ri];
        c=[-55+15*re.^2;        -55*ones(n_inh,1)];
        d=[10-6*re.^2;           4*ones(n_inh,1)];
        
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
        G = S(ind,1:ind-1);    
        G = [G,S(ind,ind+1:end)];  
        R = states(ind,:);
        S = states(1:ind-1,:);
        S = [S;states(ind+1:end,:)];
        S = S';
    else                
        re=rand(n_exc,1);          ri=rand(n_inh,1);
        a=[0.02*ones(n_exc,1);     0.02+0.08*ri];
        b=[0.2*ones(n_exc,1);      0.25-0.05*ri];
        c=[-65+15*re.^2;        -65*ones(n_inh,1)];
        d=[8-6*re.^2;           2*ones(n_inh,1)];
        S=[0.5*rand*(rand(n_exc+n_inh,n_exc)<p),  -rand(n_exc+n_inh,n_inh)];
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
        end;
        
        states = [states_exc;states_inh];
        ind = 1+floor(n_exc*rand);
        G = S(ind,1:ind-1);    
        G = [G,S(ind,ind+1:end)];
        R = states(ind,:);
        S = states(1:ind-1,:);
        S = [S;states(ind+1:end,:)];
        S = S';
    end


%==========================================================================    

%=============================DYNAMIC NETWORK==============================
elseif (mode == 3)
    
    %----------------------In-Loop Initializations-------------------------
    theta = params(1);
    q = params(2);
    tau = params(3);
    %----------------------------------------------------------------------
    
    %----------------------Create an Erdos Reny Random Graph---------------    
    n = n_exc + n_inh;
    G = erdos_reny_inhibitory(n,n_exc,n_inh,p);        
    %----------------------------------------------------------------------
                
    %------------------------Record Sample States--------------------------
    s = max(rand(1,n) < q,0);       % initial stimulus applied to n neurons            
    R = zeros(T,n);                 % recorded stimulus states             
                
    t_last = zeros(1,n);            % The last time each neuron fired
    for i = 1:T                    
        
        R(i,:) = s';                % record the current state                
        
        j = 1+floor(rand*n);        % pick a neuron to update its state
        
        denom = tau.^(0:1:i-t_last(j)-1);
        h = sum(G(j,:)*(R(i:-1:t_last(j)+1,:)')./denom);
            
        s(j) = 1*(h/n > theta);        
        if (s(j))
            t_last(j) =i;
        end
        
        
    end    
    %----------------------------------------------------------------------
    
    S = R;
%==========================================================================

%=======================ACCURATE ARTIFICIAL NEURONS========================
elseif (mode == 4)
    
    %----------------------In-Loop Initializations-------------------------
    ensmeble_count = params(1);    
    %----------------------------------------------------------------------
    
    %----------------------Create an Erdos Reny Random Graph---------------        
    G = read_graph(n_exc,n_inh,ensmeble_count);
    %----------------------------------------------------------------------
                
    %------------------------Record Sample States--------------------------
    S = read_spikes(n_exc,n_inh,ensmeble_count,T);    
    S = S';
    R = S;
    %----------------------------------------------------------------------
    
    
%==========================================================================

else
    error('Invalid state generating mode')
end