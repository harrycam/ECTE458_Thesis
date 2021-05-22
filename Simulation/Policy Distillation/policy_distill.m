%% Policy Distillation function to transfer control policies of SRL controller to improve control of Expert FLC

clear, clc;

data = load('distillation_data'); 
data = data.distillation_data;

X = data(:,1:4);
Y = data(:,5:6);

trnX = X(1:2:end,:); % Training input data set
trnY = Y(1:2:end,:); % Training output data set
vldX = X(2:2:end,:); % Validation input data set
vldY = Y(2:2:end,:); % Validation output data set

dataRange = [min(data)' max(data)'];

%% Inference rule tuning with Expert FLC as student using the SRL controller data

fisin = readfis('robot_final.fis'); % Read Expert FLC data

options = tunefisOptions('Method','particleswarm',...
    'OptimizationType','learning', ...
    'NumMaxRules',20); % Set PSO tuning options

options.MethodOptions.MaxIterations = 20;

rng('default');

fisout1 = tunefis(fisin,[],trnX,trnY,options); % Tune inference rules

%% Input MFs tuning with Expert FLC as student using the SRL controller data

[in,out,rule] = getTunableSettings(fisout1); % Get MFs and inference rules
options.OptimizationType = 'tuning';
options.Method = 'patternsearch';
options.MethodOptions.MaxIterations = 60;
options.MethodOptions.UseCompletePoll = true;

rng('default')
fisout = tunefis(fisout1,[in;out;rule],trnX,trnY,options); % Tune with patternsearch algorithm

figure
plotfis(fisout)