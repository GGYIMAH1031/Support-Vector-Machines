function accuracy = testsvm(testData, testLabel, w, b)
    dataSize = size(testData, 1);
    right = 0;
    for i = 1:dataSize
        predicted = sign(w * (testData(i,:))' + b);
        if(predicted == testLabel(i))
            right = right + 1;
        end
    end
    accuracy = right / dataSize;
end