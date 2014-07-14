function w = learnRidgeRegression(X,y,lambda)

% Implement ridge regression training here
% Inputs:
% X = N x D
% y = N x 1
% lambda = scalar
% Output:
% w = D x 1

w = ((lambda * size(X,1)) * eye(size(X, 2)) + transpose(X) * X)^-1 ...
            * transpose(X) * y;