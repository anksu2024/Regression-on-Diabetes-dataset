function w = learnOLERegression(X, y)
% The function Accepts two parameters x & y
% It calculates the weight for a given set of x and y
% Function is called two times.
% a) Training Data
% b) Test Data

% Implement OLE training here
% Inputs:
% X = N x D
% y = N x 1
% Output:
% w = D x 1

w = (transpose(X) * X)^(-1) * transpose(X) * y;