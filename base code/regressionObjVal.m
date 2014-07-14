function [error, error_grad] = regressionObjVal(w, X, y, lambda)

% compute squared error (scalar) and gradient of squared error with respect
% to w (vector) for the given data X and y and the regularization parameter
% lambda

error = (transpose(y - X * w) * (y - X * w)) ./ size(X,1) + lambda * transpose(w) * w;

error_grad = (-2 / size(X, 1)) * (transpose(X) * y - transpose(X) * X * w - lambda * size(X, 1) * w);