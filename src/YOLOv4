clc;
clear;
close all;
load('gTruth.mat');
trainingData = objectDetectorTrainingData(gTruth, 'SamplingFactor', 1, 'WriteLocation', tempdir);
inputSize = [416 416 3];
classNames = unique(trainingData.label);
numClasses = numel(classNames);
[anchors, ~] = estimateAnchorBoxes(trainingData, 9);
anchors = sortrows(anchors);
lgraph = yolov4Layers(inputSize, numClasses, anchors, 'tiny-yolov4-coco');
options = trainingOptions('sgdm', ...
'InitialLearnRate', 0.001, ...
'MaxEpochs', 20, ...
'MiniBatchSize', 8, ...
'Shuffle', 'every-epoch', ...
'VerboseFrequency', 20, ...
'Verbose', true, ...
'ExecutionEnvironment','auto', ...
'Plots', 'training-progress');
[detector, ~] = trainYOLOv4ObjectDetector(trainingData, lgraph, options);
testImage = imread('/Users/patron/Desktop/Cauliflower.jpeg');
[bboxes, scores, labels] = detect(detector, testImage);
annotatedImg = insertObjectAnnotation(testImage, 'rectangle', bboxes, cellstr(labels));
figure;
imshow(annotatedImg);
bright = imadjust(testImage, [], [], 1.2);
dark = imadjust(testImage, [], [], 0.7);
flip_bright = fliplr(bright);
flip_dark = fliplr(dark);
figure;
subplot(2,2,1); imshow(bright);
subplot(2,2,2); imshow(dark);
subplot(2,2,3); imshow(flip_bright);
subplot(2,2,4); imshow(flip_dark);
detectionResults = detect(detector, trainingData);
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, trainingData);
figure;
plot(recall, precision, 'b-', 'LineWidth', 2);
xlabel('Recall'); ylabel('Precision');
title(['Average Precision = ', sprintf('%.2f', ap)]);
axis([0 1 0.84 1]);
xticks(0:0.1:1);
yticks(0.84:0.02:1);
grid on;
