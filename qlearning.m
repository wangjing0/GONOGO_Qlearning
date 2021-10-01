%% NO-NOGO odor discrimination Simulation
close all;
% Parameters
clc;  clear all;
alpha =  .1; % learning rate
beta =   10; % 1/Boltzmann temperature,0: random, large:rational
go_bias = .1;

epsilon = 0.2;  % cost of Go, relative to drop of water reward
gamma =  .1; % laser stim,
epsilon_noise = 0.00; % noise in updating Q

%rng('default');

sigm = @(x) exp(x)./(1 + exp(x)); % sigmoid function
dsigm = @(x) exp(x)./(1 + exp(x)).^2; % derivative of sigmoid

Nmax = 0.8e3; % number of trials
Nsessions = 10;
Nmean = 10; % running mean for calculating %correct
Policy='softmax';%'greedy';%'softmax'
Trial_Type = {'GO';'GO-stim';'NOGO';'NOGO-stim'};
Odor = [[1,0,0,0];...% GO NOGO GO-stim NOGO-stim
        [0,1,0,0];...
        [0,0,1,0];...
        [0,0,0,1]];

Correct = [ [1];...% correct action
            [1];...
            [0];...
            [0]];
Reward_ = [[1-epsilon,0];...% Go
    [1-epsilon+gamma,0];...% GO-stim
    [ -epsilon,0];...      % NOGO
    [ -epsilon,0+gamma]];  % NOGO-stim

Nodor = size(Odor,1);
CPD = cumsum(1/Nodor.* ones(Nodor,1));
maxRT = nan(Nodor,Nsessions);

State = nan(Nmax,Nsessions); % Odor state
Action = nan(Nmax,Nsessions); % 
Reward = nan(Nmax,Nsessions); % recieved eward
RT = nan(Nmax,Nsessions);

QInit = [.2.*ones(Nodor,1) , 0.*ones(Nodor,1)];%+ epsilon_noise .*randn(Nodor,2) ;

figure('Position',[0 0 1000 800])
for iii=1:Nsessions
    % initialization
    [~,~,Od] = histcounts(rand(1,Nmax),CPD); 
    Od = Od +1; %Odor identity, random odor in each trial
   
    switch Policy        
        case 'softmax'
            w_ = QInit;
            w  = QInit;
            Q = nan([size(w_),Nmax]);
           
    end
    action = nan(1,Nmax);
    reward = nan(1,Nmax);
    % learning
    for i=1:Nmax
        %w_ = w_ + epsilon_noise .*randn(size(w_)); % introduce noise
        switch Policy            
            case 'softmax'
                odor = Odor( Od(i) , :);
                x_go = odor*w_(:,1)  ; x_nogo = odor*w_(:,2);
                prob_go = sigm((x_go - x_nogo + go_bias).*beta); %
                RT(i,iii) = 1./abs((x_go - x_nogo));
                a = rand(1)< prob_go;
                action(i) = a;
                if a ==1; indx_a = 1; else ;indx_a = 2; end
                if a
                    sign_a = 1;  q = (x_go);  dq = dsigm(x_go);
                else
                    sign_a = -1; q = (x_nogo); dq = dsigm(x_nogo);
                end
                
                r = Reward_(Od(i),indx_a) ;
                reward(i) = r;
                
                dw = alpha.*(r-q);
                I = epsilon_noise .*(randn(size(w_)));% introduce noise
                I(Od(i),indx_a) = I(Od(i),indx_a) + dw;
                w = w_ + I;
                Q(:,:,i) = w; %  Q value , Nodor x Naction x Ntrial
                w_ = w;
                
        end
    end
     State(:,iii) = Od;
    Action(:,iii) = action;% 0: no go or 1:go
    Reward(:,iii) = reward;
    
    % plotting
    for j=1:Nodor
        sh1(j) =subplot(Nodor,2,2*j-1);
        clear indx correct_action correct
        indx = (Od==j);
        correct_action = Correct(j);
        correct = double(action(indx) == correct_action);
        pcorrect = RunningMean_edge(correct,Nmean);
        plot(pcorrect(:,1),'-'); drawnow; hold on
        
        title(Trial_Type{j});
        ylim([-.1 1.1])
        sh2(j) =subplot(Nodor,2,2*j);
        semilogy(RT(indx,iii),'-'); hold on
        % title(['mRT = ', num2str(nanmean(RT(indx,iii)),3)])
        [~,maxRT(j,iii)] = max((RT(indx,iii)));
    end
    if iii==1
        linkaxes(sh1);
        linkaxes(sh2);
        subplot(Nodor,2,1)
        xlabel('Trial');ylabel('%Correct')
        text((Nmax/Nodor/2),.3,{['\alpha = ', num2str(alpha)];['\beta = ', num2str(beta)];['Go bias = ', num2str(go_bias)]},'Color','red','FontSize',15)
        subplot(Nodor,2,2)
        xlabel('Trial');ylabel('Response time')
    end
end
%% GO-NOGO odor discrimination Behavioral fit to Model
close all;clc;
opts = optimset('display','iter', 'PlotFcns',{@optimplotx,...
    @optimplotfval},'MaxFunEvals',1e4,'TolX',1e-6); % debugging version
opts_ = optimset('display','off','MaxFunEvals',1e3,'TolX',1e-6); % simple version

Nrepeats = 1; %number of fittings, initial the parameters at random start point
para_names = {'\alpha','\beta','go bias'};
para_truth = [alpha, beta,go_bias];
Nparas = length(para_truth);
Para_opt = nan(Nparas,Nrepeats,Nsessions);

for jjj=1:Nsessions
        state  = State(:,jjj);
        action = Action(:,jjj);
        reward = Reward(:,jjj);
    for iii=1:Nrepeats
        Para_init= abs(para_truth.*(1 + .5*randn(size(para_truth))));
        [Para_opt(:,iii,jjj),nll] = fminsearch(@(Para)GNG_loglikeli_action(Para,state,action,reward,Policy,QInit),Para_init,opts_);        
    end
end
%
figure('Position',[0 500 700 200]);
for ppp=1:Nparas
subplot(1,Nparas,ppp); hold on
bar(0,para_truth(ppp),'FaceColor','k','EdgeColor','none','FaceAlpha',.3 );
para = reshape(squeeze(Para_opt(ppp,:,:)),[],1);
plot(.1*randn(size(para)),para,'ko');
errorbar(0,nanmean(para),nanstd(para),'ko','LineWidth',2,'MarkerFaceColor','k','CapSize',0)
ylabel(para_names{ppp});
xticks([ ])
end
legend({'Ground truth';'model fit';'mean+/-sd'})
