trainData = load('diabetic-train.mat');
testData = load('diabetic-test.mat');
scaledTrain = (trainData.x-min(trainData.x(:))) ./ (max(trainData.x(:))-min(trainData.x(:)));
scaledTest = (testData.x-min(testData.x(:))) ./ (max(testData.x(:))-min(testData.x(:)));
trainData.y(trainData.y==0) = -1;
testData.y(testData.y==0) = -1;
C = [4^-6 4^-5 4^-4 4^-3 4^-2 0.25 1 4 16];  
disp('Value of C | Accuracy | Time');
valAccuracy = [];
for i = 1:length(C)
    [w, b, time] = trainsvm(scaledTrain, trainData.y, C(i));
    valAccuracy(i) = cross_validation(scaledTrain, trainData.y, w, b);
    disp([C(i) valAccuracy(i) time]);
end
[~, Cindex] = max(valAccuracy(:));
optC = C(Cindex);
[w, b, ~] = trainsvm(scaledTrain, trainData.y, optC);
testAccuracy = testsvm(scaledTest, testData.y, w, b);

options = '-v 3 -c ';
for i = 1:length(C)
    startTime = tic;
    svmModel = svmtrain(scaledTrain, trainData.y, [options num2str(C(i))]);
    timeElapsed = toc(startTime)/5;
end
