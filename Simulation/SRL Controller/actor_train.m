%% Train DNN actor for AR navigation using training sets and SL

clear, clc

numObs = 6; 
numAct = 2;

% Create the actor network layers

actorLayerSizes2 = [16 16];
actorNetwork2 = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(actorLayerSizes(1), 'Name', 'ActorFC1', ...
            'Weights',2/sqrt(numObs)*(rand(actorLayerSizes(1),numObs)-0.5), ... 
            'Bias',2/sqrt(numObs)*(rand(actorLayerSizes(1),1)-0.5))
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
    regressionLayer('Name','RegressionOutput')
    ];

%% Import Training Data

training_data = load('training_data\training.mat');
data = training_data.training;
%data = data(1:24255,:);
totalRows = size(data,1);

%% Create Sets

validationSplitPercent = 0.1;
numValidationDataRows = floor(validationSplitPercent*totalRows);

testSplitPercent = 0.05;
numTestDataRows = floor(testSplitPercent*totalRows);

randomIdx = randperm(totalRows,numValidationDataRows + numTestDataRows);
randomData = data(randomIdx,:);

validationData = randomData(1:numValidationDataRows,:);
testData = randomData(numValidationDataRows + 1:end,:);

trainDataIdx = setdiff(1:totalRows,randomIdx);
trainData = data(trainDataIdx,:);

numTrainDataRows = size(trainData,1);
shuffleIdx = randperm(numTrainDataRows);
shuffledTrainData = trainData(shuffleIdx,:);

%% Reshape Sets

numObservations = 6; 
numActions = 2;

trainInput = reshape(shuffledTrainData(:,1:6)',[numObservations 1 1 numTrainDataRows]);
trainOutput = reshape(shuffledTrainData(:,7:8)',[1 1 numActions numTrainDataRows]);

validationInput = reshape(validationData(:,1:6)',[numObservations 1 1 numValidationDataRows]);
validationOutput = reshape(validationData(:,7:8)',[1 1 numActions numValidationDataRows]);
validationCellArray = {validationInput,validationOutput};

testDataInput = reshape(testData(:,1:6)',[numObservations 1 1 numTestDataRows]);
testDataOutput = testData(:,7:8);

%% Training Neural Network

options = trainingOptions('adam', ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'Shuffle','every-epoch', ...
    'MaxEpochs', 60, ...
    'MiniBatchSize',512, ...
    'ValidationData',validationCellArray, ...
    'InitialLearnRate',1e-3, ...
    'GradientThresholdMethod','absolute-value', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',10, ...
    'Epsilon',1e-8);

ActorNetObj2 = trainNetwork(trainInput,trainOutput,actorNetwork2,options);