function nLL = GNG_loglikeli_action(Para,state,action,reward, Policy,QInit)
% negative loglikeli function of the GO-NOGO task.
alpha = Para(1); % learning rate for chosen action
beta = Para(2); % 1/temperature
go_bias =  Para(3); % go bias

exp_sum = @(x) sum(exp(x)); % sum of exp function

is_nan = ~(isnan(state) | isnan(action) | isnan(reward));
state = state(is_nan);
action = action(is_nan);% action = 0: nogo 1: go
reward = reward(is_nan);

Nmax  = length(state);
LL = nan(Nmax,1); % loglikeli for each trial
[Nstim,Naction] = size(QInit);
  Q = nan(Nstim,Naction,Nmax); 
switch Policy
    case 'softmax'
             
        for t=1:Nmax
            if t==1
                Q(:,:,t) = QInit;
            end
            ind_state = state(t);
            ind_action = 2-action(t);
            qt = squeeze(Q(ind_state,:,t)) + [go_bias,0];
            LL(t) = beta.*qt(ind_action) - log(exp_sum(beta.*qt));
            dr = reward(t) - Q(ind_state,ind_action,t);% RPE
            
            if t< Nmax
                Q(:,:,t+1) =  Q(:,:,t);
                Q(ind_state,ind_action,t+1) = Q(ind_state,ind_action,t+1) + alpha.*dr;
                
            end
        end
                
    otherwise
       
        disp('Error! LL does not exist.')
end     

     nLL = -nanmean(LL);
end
