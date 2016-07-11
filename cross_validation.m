function accuracyVal = cross_validation(data, labels, w, b)
    dataSize = size(data,1);
    num_of_folds = 3;
    x = [];
    y = [];
    nps = dataSize/num_of_folds;
    for i = 1:nps:dataSize
        x = cat(3, x, data(i:i+nps-1,:));
        y = [y labels(i:i+nps-1)];
    end
    dataSets = size(x,3);
    accuracy = [];
    for i = 1:dataSets
        accuracy = [accuracy testsvm(x(:,:,i),y(:,i),w,b)];
    end
    accuracyVal = mean(accuracy);
end

