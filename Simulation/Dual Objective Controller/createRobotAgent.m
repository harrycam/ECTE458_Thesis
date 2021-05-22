% Overall function to create and train the Dual Objective DPPG learning agent in the
% environment.
%% SET UP ENVIRONMENT

clc
close all

global x_pos y_pos ang_start counter use_fuzzy

Ts = 0.025; % Agent sample time
Tf = 45;    % Simulation end time

counter = 0;
use_fuzzy = 1;

% Speedup options
useFastRestart = false;

robot = readfis('robot_final.fis');

% Create the avoidance observation info
numObsAvoidance = 4;
observationInfoAvoidance = rlNumericSpec([numObsAvoidance 1]);
observationInfoAvoidance.Name = 'observations';

% Create the target observation info
numObsTarget = 4;
observationInfoTarget = rlNumericSpec([numObsTarget 1]);
observationInfoTarget.Name = 'observations';

% create the action info
numAct = 2;
actionInfoAvoidance = rlNumericSpec([numAct 1],'LowerLimit',-2,'UpperLimit', 2);
actionInfoAvoidance.Name = 'wheel_velocity';

actionInfoTarget = rlNumericSpec([numAct 1],'LowerLimit',-2,'UpperLimit', 2);
actionInfoTarget.Name = 'wheel_velocity';

observationInfo = {observationInfoAvoidance,observationInfoTarget};
actionInfoTotal = {actionInfoAvoidance,actionInfoTarget};

% Environment
mdl = "RoboBlockRLDO";
load_system(mdl);
blk = mdl + ["/RL Avoidance", "/RL Target"];
env = rlSimulinkEnv(mdl,blk,observationInfo,actionInfoTotal);
env.ResetFcn = @(in)ResetFcn(in);
%function to allow changing of parameters on reset - investigate this

if ~useFastRestart
   env.UseFastRestart = 'off';
end
%% CREATE NEURAL NETWORKS
createDDPGNetworks; % Create critic and both target and avoidance actor DNNs
                     
%% CREATE AND TRAIN AGENT
createDDPGOptions;
agentTarget = rlDDPGAgent(actorTarget,criticTarget,agentOptions);
agentAvoidance = rlDDPGAgent(actorAvoidance,criticAvoidance,agentOptions);
trainingResults = train([agentAvoidance, agentTarget],env,trainingOptions);