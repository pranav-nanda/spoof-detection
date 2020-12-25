%% Transfer learning
net = resnet50();   % import the pre trained network along with the weights as a DAGNetwork
lgraph = layerGraph(net);   % convert the imported net to a layer graph object for ease of manipulation of layers
newConnectedLayer = fullyConnectedLayer(2,'Name','new_fc');  % create a new fully connected layer with two outputs
lgraph = replaceLayer(lgraph,'fc1000',newConnectedLayer);   % replace the existing fully connected layer with the new layer
newClassificationLayer = classificationLayer('Name','new_classoutput');  % similarly create a new classification layer
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassificationLayer); % Replace the existing layer

%% Set training options
options = trainingOptions('adam',...  
    'MiniBatchSize',256,...
    'MaxEpochs',3,...
    'InitialLearnRate',1e-4,...
    'L2Regularization',0.0001,...
    'Verbose',1,...
    'ExecutionEnvironment','gpu',...
    'Plots','training-progress');
rng default

%% Load the train and dev images into the workspace as Image Data stores
trainfolder = fullfile('..','New folder (6)') % replace folder name to "LH_train","H_train" or "mfcc_train"  for the other sets of features
trainImages = imageDatastore(trainfolder,'IncludeSubfolders',true,...
    'FileExtensions','.tif',...
    'LabelSource','foldernames');

devfolder = fullfile('..','LL_dev')  % replace folder name to "LH_dev","H_dev" or "mfcc_dev"  for the other sets of features
devImages = imageDatastore(devfolder,'IncludeSubfolders',true,...
    'FileExtensions','.tif',...
    'LabelSource','foldernames');

%% Train the Network
LA_LL_net = trainNetwork(trainImages,lgraph,options);  % begin training the network on the LL feature
% LA_LH_net = trainNetwork(trainImages,lgraph,options); % begin training the network on the LH feature
% LA_H_net = trainNetwork(trainImages,lgraph,options);  % begin training the network on the H feature
% LA_mfcc_net = trainNetwork(trainImages,lgraph,options); % begin training the network on the mfcc feature

%% Compute scores
scores_LA_LL = predict(LA_LL_net,devImages);  % compute the scores on the development set for the LL feature
% scores_LA_LH = predict(LA_LH_net,devImages); % compute the scores on the development set for the LH feature
% scores_LA_H = predict(LA_H_net,devImages);   % compute the scores on the development set for the H feature
% scores_LA_mfcc = predict(LA_mfcc_net,devImages); % compute the scores on the development set for the mfcc feature