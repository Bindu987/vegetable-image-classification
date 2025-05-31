clc;
clear;
close all;
datasetPath = 'C:\Users\mm\Desktop\Vegetable Images';
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
allowedClasses = ["Bottle_Gourd", "Brinjal", "Broccoli", "Carrot", "Cauliflower", "Cucumber", "Potato", "Radish", "Pumpkin"];
imds = subset(imds, ismember(imds.Labels, allowedClasses));
imds = splitEachLabel(imds, 1000, 'randomized');
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));
net = squeezenet;
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
newConv = convolution2dLayer(1, numClasses, 'Name', 'new_conv');
newConv.WeightsInitializer = 'he';
newConv.BiasInitializer = 'zeros';
newPool = globalAveragePooling2dLayer('Name', 'new_pool');
newSoftmax = softmaxLayer('Name', 'new_softmax');
newClass = classificationLayer('Name', 'new_output');
lgraph = replaceLayer(lgraph, 'conv10', newConv);
lgraph = replaceLayer(lgraph, 'pool10', newPool);
lgraph = replaceLayer(lgraph, 'prob', newSoftmax);
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
trainedSqueeze = trainNetwork(augTrain, lgraph, options);
YPred = classify(trainedSqueeze, augVal);
YTrue = imdsVal.Labels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('SqueezeNet Validation Accuracy: %.5f%%\n', accuracy * 100)
figure;
confusionchart(YTrue, YPred);
title('Confusion Matrix - SqueezeNet (Vegetable Classification)');
xlabel('Predicted Class');
ylabel('True Class');
