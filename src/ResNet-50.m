clc;
clear;
close all;
datasetPath = 'C:\Users\mm\Desktop\Vegetable Images';
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds = splitEachLabel(imds, 1000, 'randomized');
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));
net = resnet50;
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});
newLayers = [
fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
softmaxLayer('Name', 'new_softmax')
classificationLayer('Name', 'new_class')];
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'new_fc');
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal = augmentedImageDatastore(inputSize, imdsVal);
options = trainingOptions('sgdm', ...
'InitialLearnRate', 0.001, ...
'MaxEpochs', 20, ...
'MiniBatchSize', 32, ...
'Shuffle', 'every-epoch', ...
'ValidationData', augVal, ...
'ValidationFrequency', 30, ...
'Verbose', true, ...
'Plots', 'training-progress');
trainedNet = trainNetwork(augTrain, lgraph, options);
YPred = classify(trainedNet, augVal);
YTrue = imdsVal.Labels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Validation Accuracy: %.5f%%\n', accuracy * 100);
figure;
confusionchart(YTrue, YPred);
title('Confusion Matrix - ResNet-50 (Vegetable Classification)');
xlabel('Predicted Class');
ylabel('True Class');
