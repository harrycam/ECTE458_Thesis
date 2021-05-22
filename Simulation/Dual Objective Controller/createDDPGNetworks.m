% Code to create the actor and critic DNNs for the DDPG algorithm

ObsTotal = env.getObservationInfo;
ActsTotal = env.getActionInfo;

%% CRITIC Avoidance
% Create the critic avoidance network layers

criticLayerSizes = [16 16];
statePath = [
    imageInputLayer([numObsAvoidance 1 1],'Normalization','none','Name', 'observation')
    fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticStateFC1', ... 
            'Weights',2/sqrt(numObsAvoidance)*(rand(criticLayerSizes(1),numObsAvoidance)-0.5), ...
            'Bias',2/sqrt(numObsAvoidance)*(rand(criticLayerSizes(1),1)-0.5))
    reluLayer('Name','CriticStateRelu1')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticStateFC2', ...
            'Weights',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ... 
            'Bias',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),1)-0.5))
    ];
actionPath = [
    imageInputLayer([numAct 1 1],'Normalization','none', 'Name', 'action')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticActionFC1', ...
            'Weights',2/sqrt(numAct)*(rand(criticLayerSizes(2),numAct)-0.5), ... 
            'Bias',2/sqrt(numAct)*(rand(criticLayerSizes(2),1)-0.5))
    ];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1, 'Name', 'CriticOutput',...
            'Weights',2*5e-3*(rand(1,criticLayerSizes(2))-0.5), ...
            'Bias',2*5e-3*(rand(1,1)-0.5))
    ];
% Connect the layer graph
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

% Create critic representation
criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-4, ... 
                                        'GradientThreshold',1,'L2RegularizationFactor',2e-4);
                          
criticAvoidance = rlQValueRepresentation(criticNetwork,ObsTotal{1},ActsTotal{1}, ...
    'Observation',{'observation'}, ...
    'Action',{'action'}, ...
    criticOptions);

%% CRITIC Target

% Create the critic target network layers

criticLayerSizes = [16 16];
statePath = [
    imageInputLayer([numObsTarget 1 1],'Normalization','none','Name', 'observation')
    fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticStateFC1', ... 
            'Weights',2/sqrt(numObsTarget)*(rand(criticLayerSizes(1),numObsTarget)-0.5), ...
            'Bias',2/sqrt(numObsTarget)*(rand(criticLayerSizes(1),1)-0.5))
    reluLayer('Name','CriticStateRelu1')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticStateFC2', ...
            'Weights',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ... 
            'Bias',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),1)-0.5))
    ];
actionPath = [
    imageInputLayer([numAct 1 1],'Normalization','none', 'Name', 'action')
    fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticActionFC1', ...
            'Weights',2/sqrt(numAct)*(rand(criticLayerSizes(2),numAct)-0.5), ... 
            'Bias',2/sqrt(numAct)*(rand(criticLayerSizes(2),1)-0.5))
    ];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1, 'Name', 'CriticOutput',...
            'Weights',2*5e-3*(rand(1,criticLayerSizes(2))-0.5), ...
            'Bias',2*5e-3*(rand(1,1)-0.5))
    ];
% Connect the layer graph
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

% Create critic representation
criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-4, ... 
                                        'GradientThreshold',1,'L2RegularizationFactor',2e-4);
                          
criticTarget = rlQValueRepresentation(criticNetwork,ObsTotal{2},ActsTotal{2}, ...
    'Observation',{'observation'}, ...
    'Action',{'action'}, ...
    criticOptions);

%% Create Avoidance Actor

load('ActorSupervised3.mat','ActorNetObj');

actorLayerSizes = [16 16];
actorNetwork = [
    imageInputLayer([numObsAvoidance 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(actorLayerSizes(1), 'Name', 'ActorFC1', ...
            'Weights',2/sqrt(numObsAvoidance)*(rand(actorLayerSizes(1),numObsAvoidance)-0.5), ... 
            'Bias',2/sqrt(numObsAvoidance)*(rand(actorLayerSizes(1),1)-0.5))
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(actorLayerSizes(2), 'Name', 'ActorFC2', ... 
            'Weights',2/sqrt(actorLayerSizes(1))*(rand(actorLayerSizes(2),actorLayerSizes(1))-0.5), ... 
            'Bias',2/sqrt(actorLayerSizes(1))*(rand(actorLayerSizes(2),1)-0.5))
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(numAct, 'Name', 'ActorFC3', ... 
            'Weights',2*5e-3*(rand(numAct,actorLayerSizes(2))-0.5), ... 
            'Bias',2*5e-3*(rand(numAct,1)-0.5))                       
    tanhLayer('Name','ActorTanh1')
    scalingLayer('Name','Scale1','Scale',2)
    ];

% Create actor representation
actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3, ...
                                       'GradientThreshold',1,'L2RegularizationFactor',2e-4);

actorAvoidance = rlDeterministicActorRepresentation(actorNetwork,ObsTotal{1},ActsTotal{1}, ... 
                         'Observation',{'observation'}, ...
                         'Action',{'Scale1'}, ...
                         actorOptions);           
%% Create Target Actor

actorLayerSizes = [16 16];
actorNetwork = [
    imageInputLayer([numObsTarget 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(actorLayerSizes(1), 'Name', 'ActorFC1', ...
            'Weights',2/sqrt(numObsTarget)*(rand(actorLayerSizes(1),numObsTarget)-0.5), ... 
            'Bias',2/sqrt(numObsTarget)*(rand(actorLayerSizes(1),1)-0.5))
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(actorLayerSizes(2), 'Name', 'ActorFC2', ... 
            'Weights',2/sqrt(actorLayerSizes(1))*(rand(actorLayerSizes(2),actorLayerSizes(1))-0.5), ... 
            'Bias',2/sqrt(actorLayerSizes(1))*(rand(actorLayerSizes(2),1)-0.5))
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(numAct, 'Name', 'ActorFC3', ... 
            'Weights',2*5e-3*(rand(numAct,actorLayerSizes(2))-0.5), ... 
            'Bias',2*5e-3*(rand(numAct,1)-0.5))                       
    tanhLayer('Name','ActorTanh1')
    scalingLayer('Name','Scale1','Scale',2)
    ];

% Create actor representation
actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-2, ...
                                       'GradientThreshold',1,'L2RegularizationFactor',2e-4);

actorTarget = rlDeterministicActorRepresentation(actorNetwork,ObsTotal{2},ActsTotal{2}, ... 
                         'Observation',{'observation'}, ...
                         'Action',{'Scale1'}, ...
                         actorOptions);