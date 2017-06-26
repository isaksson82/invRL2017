%% function calculating -logLH for one subject given parameters
% for use with Matlab's fmincon function
function [LLHi] = fnMinLLH_iRL_pseudo(parameters)

global condition SubjFolder fitOutput;

% load behavior
if condition==1
    input = load([SubjFolder filesep 'banditmat_sim_DONE.mat']);
    
elseif condition==2
    input = load([SubjFolder filesep 'banditmat_dis_DONE.mat']);
end

bandit      = input.banditmat(:,[3 14 15 17 18]); 

% chosen bandit # | unchosen bandit #
A_disp      = bandit(:,[2 3]); 

% onset [slot machine | feedback]
onsets      = bandit(:,[4 5]);

% trialtype ( 1=participant | 2=agent)
oth_self    = bandit(:,1);

% number of trials
Ntrials     = length(bandit); 

% some initialisations
subjtrials  = 0; 
subjnonerror= 0;
LLHi        = 0;
Psum        = 0;
fit         = [];
agentTrials = 0;

% participant's fixed food preferences (linear magnitude)
goalValues  = [1 2 3];

%%
% ---------------------------------------
%          4 FIT PARAMETERS
% ---------------------------------------

updatePriors        = [parameters(1) parameters(2) parameters(3)];

% participant softmax temperature
betaO   = parameters(4); 

% starting PROBABILITY DISTR. for other's bandits 1 to 3.
Qall_o  = [  1/3 1/3 1/3;...
            1/3 1/3 1/3;...
            1/3 1/3 1/3 ];
              
updatePriorsunch    = updatePriors(end:-1:1);


%%
% ---------------------------------------
%                   FIT
% ---------------------------------------

% loop over trials
for time=1:Ntrials
    
    a                       = A_disp(time,:); % a(1) = chosen bandit, a(2) = unchosen bandit
    a_nodisp                = 6 - a(1) - a(2);% bandit not displayed
    o_s                     = oth_self(time); % 1=subject, 2=sim/dis agent
    
    if a(1)>0 % if no error
        
        if o_s == 2 % agent's turn
            %%
            % ---------------------------------------
            %    AGENT'S TURN: Outcome prediction
            % ---------------------------------------
            
            agentTrials = agentTrials+1;
            
            % distributions of bandit a(1), resp. a(2)
            Qo      = Qall_o(a,:);  
            
            %%
            % ---------------------------------------
            %    AGENT'S TURN: bandit update
            % ---------------------------------------
            
            % update of slot machine distributions, given choice of agent
            QoUp(1,:) = (updatePriors.* Qo(1,:) )./sum(  updatePriors.*Qo(1,:) ); % chosen
            QoUp(2,:) = (updatePriorsunch.*Qo(2,:))./sum(  updatePriorsunch.* Qo(2,:)); % unchosen
            
            Qall_o(a(1),:)  = QoUp(1,:);
            Qall_o(a(2),:)  = QoUp(2,:);
            
            
        elseif o_s==1
            %%
            % ---------------------------------------
            %              SELF TURN
            % ---------------------------------------
           
            subjtrials      = subjtrials+1;
            subjnonerror    = subjnonerror + 1;
            
            % distributions of bandit a(1), resp. a(2)
            Qall_s      = Qall_o;
            Qs          = Qall_s(a,:); 
            Qs_nodisp   =  Qall_s(a_nodisp,:);
           
            % normalized slot machine values
            EVchosen    = sum(Qs(1,:).*goalValues); % self EVs for self choice
            EVunchosen  = sum(Qs(2,:).*goalValues); % self EVs for self not-choice
            EV_nodisp   = sum(Qs_nodisp.*goalValues);
            allEV       = [EVchosen EVunchosen EV_nodisp]; % i.e. allEV(1) ~ a(1); allEV(2) ~ a(2);
            allEV       = allEV./sum(allEV);
            EVchosen    = allEV(1);
            EVunchosen  = allEV(2);
            
            % softmax decision
            selfChoiceP = Psoftmax(EVchosen, EVunchosen, -betaO);
            
            LLHi = LLHi-log(selfChoiceP);
            Psum = selfChoiceP+Psum;
            
        end
    else
        % participant error
        subjtrials=subjtrials+1;
    end
end

fitOutput       = fit;

end

