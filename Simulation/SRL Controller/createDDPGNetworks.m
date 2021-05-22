% Code to create the actor and critic DNNs for the DDPG algorithm

%% CRITIC
% Create the critic network layers

criticLayerSizes = [200 200];
statePath = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name', 'observation')
    fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticStateFC1', ... 
            'Weights',2/sqrt(numObs)*(rand(criticLayerSizes(1),numObs)-0.5), ...
            'Bias',2/sqrt(numObs)*(rand(criticLayerSizes(1),1)-0.5))
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
                                                           
critic = rlQValueRepresentation(criticNetwork,env.getObservationInfo,env.getActionInfo, ...
    'Observation',{'observation'}, ...
    'Action',{'action'}, ...
    criticOptions);


%% Create Pretrained Actor

load('ActorSupervised3.mat','ActorNetObj'); 

% Create actor representation
actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-6, ...
                                       'GradientThreshold',1,'L2RegularizationFactor',2e-4);

actor = rlDeterministicActorRepresentation(ActorNetObj,env.getObservationInfo,env.getActionInfo, ... 
                         'Observation',ActorNetObj.InputNames, ...
                         'Action',{'Scale1'}, ...
                         actorOptions);