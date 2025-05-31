clc;
clear;
close all;
datasetPath = 'C:\Users\mm\Desktop\Vegetable Images';
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds = splitEachLabel(imds, 1000, 'randomized');
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));
inputSize = [224 224 3];
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal = augmentedImageDatastore(inputSize, imdsVal);
layers = [
imageInputLayer(inputSize, 'Name', 'input')
convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
batchNormalizationLayer('Name', 'bn1')
reluLayer('Name', 'relu1')
maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
batchNormalizationLayer('Name', 'bn2')
reluLayer('Name', 'relu2')
maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
batchNormalizationLayer('Name', 'bn3')
reluLayer('Name', 'relu3')
maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
fullyConnectedLayer(numClasses, 'Name', 'fc')
softmaxLayer('Name', 'softmax')
classificationLayer('Name', 'output')
];
options = trainingOptions('sgdm', ...
'InitialLearnRate', 0.001, ...
'MaxEpochs', 20, ...
'MiniBatchSize', 32, ...
'Shuffle', 'every-epoch', ...
'ValidationData', augVal, ...
'ValidationFrequency', 30, ...
'Verbose', true, ...
'Plots', 'training-progress');
netCustom = trainNetwork(augTrain, layers, options);
YPred = classify(netCustom, augVal);
YTrue = imdsVal.Labels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Custom DCNN Validation Accuracy: %.5f%%\n', accuracy * 100);
figure;
confusionchart(YTrue, YPred);
title('Confusion Matrix - Custom DCNN (Vegetable Classification)');
xlabel('Predicted Class');
ylabel('True Class');
