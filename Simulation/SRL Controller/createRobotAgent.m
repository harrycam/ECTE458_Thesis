% Overall function to create and train the DPPG learning agent in the
% environment. The Expert FLC can be used for additional observation
% initially in RL, but this has been disabled
%% SET UP ENVIRONMENT

clc
close all

global x_pos y_pos ang_start counter use_fuzzy

Ts = 0.025; % Agent sample time
Tf = 30;    % Simulation end time
counter = 0;
use_fuzzy = 1;

% Speedup options
useFastRestart = false;

robot = readfis('robot_final.fis');

% Create the observation info
numObs = 6;
observationInfo = rlNumericSpec([numObs 1]);
observationInfo.Name = 'observations';

% create the action info
numAct = 2;
actionInfo = rlNumericSpec([numAct 1],'LowerLimit',-2,'UpperLimit', 2);
actionInfo.Name = 'wheel_velocity';
% Environment

mdl = 'RoboBlockRL';
load_system(mdl);
blk = [mdl,'/RL Agent'];
env = rlSimulinkEnv(mdl,blk,observationInfo,actionInfo);
env.ResetFcn = @(in)ResetFcn(in); % Function to allow position and angle to be set each episode

if ~useFastRestart
   env.UseFastRestart = 'off';
end
%% CREATE NEURAL NETWORKS
createDDPGNetworks; % Create the actor anf critic DNN networks
                     
%% CREATE AND TRAIN AGENT
createDDPGOptions; % Set options for training
agent = rlDDPGAgent(actor,critic,agentOptions); % Create DDPG learning agent
trainingResults = train(agent,env,trainingOptions) % Train the DDPG learning agent