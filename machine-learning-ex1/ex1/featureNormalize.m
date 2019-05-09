function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));    % number of features (# of columns)
sigma = zeros(1, size(X, 2)); % number of features (# of columns)



num_of_features = size(X, 2);
for feature_index = 1:num_of_features 
  sigma(feature_index) = std(X(:,feature_index));
  mu(feature_index) = mean(X(:,feature_index));
  
  X_norm(:,feature_index) = (X(:,feature_index) - mu(feature_index))./sigma(feature_index);
end




% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       









% ============================================================

end
