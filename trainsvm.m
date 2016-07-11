function [w, b, telapsed] = trainsvm(trainData, trainLabel, C)
    tstart = tic;
    [x, y] = size(trainData);
    K = trainData * trainData';
    H = (trainLabel * trainLabel') .* K + 1e-5*eye(x);
    f = ones(x,1);
    lb = zeros(x,1);
    ub = repmat(C,x,1);
    Aeq = trainLabel';
    beq = 0;
    options = optimset('Display', 'off');
    alpha = quadprog(H, -f, [], [], Aeq, beq, lb, ub, [], options);
    tmp = sum(repmat(alpha .* trainLabel, 1, x) .* K, 1)';
    bpos = find(alpha > 1e-6);
    b = mean(trainLabel(bpos) - tmp(bpos));
    w = sum(repmat(alpha .* trainLabel, 1, y) .* trainData, 1);
    telapsed = toc(tstart);
end