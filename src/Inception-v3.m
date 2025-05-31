clc;
clear;
close all;
datasetPath = 'C:\Users\mm\Desktop\Vegetable Images';
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
allowedClasses = ["Bottle_Gourd", "Brinjal", "Broccoli", "Carrot", "Cauliflower", "Cucumber", "Potato",
"Radish", "Pumpkin"];
imds = subset(imds, ismember(imds.Labels, allowedClasses));
imds = splitEachLabel(imds, 1000, 'randomized');
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));
net = inceptionv3;
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
newFc = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
lgraph = replaceLayer(lgraph, 'predictions', newFc);
newClass = classificationLayer('Name', 'new_class');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClass);
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal = augmentedImageDatastore(inputSize, imdsVal);
totalTrainImages = numel(imdsTrain.Labels);
miniBatchSize = floor(totalTrainImages / 500);
fprintf('MiniBatch Size set to %d images.\n', miniBatchSize);
options = trainingOptions('adam', ...
'InitialLearnRate', 1e-4, ...
'MaxEpochs', 20, ...
'MiniBatchSize', miniBatchSize, ...
'Shuffle', 'every-epoch', ...
'ValidationData', augVal, ...
'ValidationFrequency', 30, ...
'Verbose', true, ...
'Plots', 'training-progress');
trainedInception = trainNetwork(augTrain, lgraph, options);
YPred = classify(trainedInception, augVal);
YTrue = imdsVal.Labels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Inception-v3 Validation Accuracy: %.5f%%\n', accuracy * 100);
figure;
confusionchart(YTrue, YPred);
title('Confusion Matrix - Inception-v3 (Vegetable Classification)');
xlabel('Predicted Class');
ylabel('True Class');
