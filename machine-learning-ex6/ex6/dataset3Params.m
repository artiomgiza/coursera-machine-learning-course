function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

%% ====================== YOUR CODE HERE ======================
%% Instructions: Fill in this function to return the optimal C and sigma
%%               learning parameters found using the cross validation set.
%%               You can use svmPredict to predict the labels on the cross
%%               validation set. For example, 
%%                   predictions = svmPredict(model, Xval);
%%               will return the predictions on the cross validation set.
%%
%%  Note: You can compute the prediction error using 
%%        mean(double(predictions ~= yval))
%


% UNCOMMENT TO UTRAIN

% FROM HERE ===================================================
% FROM HERE ===================================================
% FROM HERE ===================================================

%res = zeros(64,3);
%i = 0;
%
%for C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
%  for sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
%    i = i + 1;
%    
%    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
%    
%    predictions = svmPredict(model, Xval);
%    
%    err = mean(double(predictions ~= yval));
%    
%    fprintf(['%d ==> C = %f, sigma = %f ---> err: %f \n'], i, C, sigma, err);
%
%    res(i,1) = C;
%    res(i,2) = sigma;
%    res(i,3) = err;
%  end
%end
%
%
%[minval, row] = min(res(:,3));
%
%C = res(row,1);
%sigma = res(row,2);
%
%fprintf(['RES: C = %f, sigma = %f ---> minval: %f (row: %d)\n'], C, sigma, minval, row);

% TO HERE ===================================================
% TO HERE ===================================================
% TO HERE ===================================================


%% best res:
    C = 1.000000
    sigma = 0.100000
    
% =========================================================================

end
