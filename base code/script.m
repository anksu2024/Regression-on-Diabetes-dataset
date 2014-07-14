%%
%% PROJECT 2: MACHINE LEARNING - REGRESSION
%%% TEAM MEMBERS : 
%%%   ANKIT SARRAF
%%%   RAHUL SINGH
%%%   KARTHICK KRISHNA VENKATAKRISHAN
%%%   

%% CLEAR WORKSPACE AND COMMAND WINDOW
clear all;
clc;

%% LOAD THE DATA
load diabetes;
% x_train_i is the training data with bias included
x_train_i = [ones(size(x_train,1),1) x_train];
% x_test_i is the test data with bias included
x_test_i = [ones(size(x_test,1),1) x_test];

%% FILL CODE FOR PROBLEM 1 %%
% PART 1: Linear regression without intercept
% w is the weight learnt for the training data without intercept
tic;
w = learnOLERegression(x_train, y_train);
% Error observed in the prediction of the train data
error_train = sqrt(transpose(y_train - x_train * w) * ...
                            (y_train - x_train * w));
% Error observed in the prediction of the test data
error_test  = sqrt(transpose(y_test - x_test * w) * ...
                            (y_test - x_test * w));

% PART 2: Linear regression with intercept
% wi is the weight learnt for the training data with intercept
wi = learnOLERegression(x_train_i, y_train);
% Error observed in the prediction of the train data (with bias)
error_train_i = sqrt(transpose(y_train - x_train_i * wi) * ...
                                (y_train - x_train_i * wi));
% Error observed in the prediction of the test data (with bias)
error_test_i = sqrt(transpose(y_test - x_test_i * wi) * ...
                                (y_test - x_test_i * wi));
toc;
% CONCLUSION:
% Error observed in the data with intercept (bias) 
% is less than the
% Error observed in the data without intercept (bias)
%%% END PROBLEM 1 CODE %%%


%% FILL CODE FOR PROBLEM 2 %%
% ridge regression using least squares - minimization
% For this use only the data with Bias
tic;
lambdas = 0:0.00001:0.001;
train_errors = zeros(length(lambdas), 1);
test_errors = zeros(length(lambdas), 1);

for i = 1:length(lambdas)
    lambda = lambdas(i);
    wRidge = learnRidgeRegression(x_train_i, y_train, lambda);
    train_errors(i) = sqrt(transpose(y_train - x_train_i * wRidge) * ...
                                    (y_train - x_train_i * wRidge));
    test_errors(i) = sqrt(transpose(y_test - x_test_i * wRidge) * ...
                                    (y_test - x_test_i * wRidge));
end

figure;
plot([train_errors test_errors]);
legend('Training Error', 'Testing Error');

% Find the value of lambda for Minimum Error
[~, minIndex] = min(test_errors);
lambda_optimal = lambdas(minIndex);
toc;
% The Graph observed for the Part 2 is because of Bias Variance Trade off
%%% END PROBLEM 2 CODE %%%
return;
%% BEGIN PROBLEM 3 CODE
% ridge regression using gradient descent - see handouts (lecture 21 p5) or
% http://cs229.stanford.edu/notes/cs229-notes1.pdf (page 11)
tic;
initialWeights = zeros(65, 1);
% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 500);

% define the objective function
% Range of lambda is to be considered as 0 to 0.001
% To observe the proper graph
lambdas = 0:0.00001:0.001;
train_errors_3 = zeros(length(lambdas), 1);
test_errors_3 = zeros(length(lambdas), 1);

% run ridge regression training with fmincg
for i = 1:length(lambdas)
    lambda = lambdas(i);
    objFunction = @(params) regressionObjVal(params, x_train_i, ...
                                        y_train, lambda);
    w = fmincg(objFunction, initialWeights, options);

    % fill code here for prediction and computing errors
    train_errors_3(i) = sqrt(transpose(y_train - x_train_i * w) * ...
                            (y_train - x_train_i * w));
    test_errors_3(i) = sqrt(transpose(y_test - x_test_i * w) * ...
                            (y_test - x_test_i * w));
end

% Find the value of lambda for Minimum Error
[minVal, minIndex] = min(test_errors);
lambda_optimal_3 = lambdas(minIndex);

figure;
plot([train_errors_3 test_errors_3]);
legend('Training Error','Testing Error');
toc;
% The Graph observed for the Part 2 is because of Bias Variance Trade off
%%% END PROBLEM 3 CODE

%% BEGIN  PROBLEM 4 CODE
%%% Non - Linear Regression

% using variable number 3 only
tic;
x_train = x_train(:, 3);
x_test = x_test(:, 3);
train_errors_4a = zeros(length(7), 1);
test_errors_4a = zeros(length(7), 1);

% no regularization
lambda = 0;
for d = 0 : 6
    x_train_n = mapNonLinear(x_train, d);
    x_test_n = mapNonLinear(x_test, d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);

    % fill code here for prediction and computing errors
    train_errors_4a(d+1) = sqrt(transpose(y_train - x_train_n * w) * ...
                                        (y_train - x_train_n * w));
    test_errors_4a(d+1) = sqrt(transpose(y_test - x_test_n * w) * ...
                                        (y_test - x_test_n * w));
end

[~, minErrIndexTr] = min(train_errors_4a);
[~, minErrIndexTe] = min(test_errors_4a);
fprintf('Lambda = 0\n');
fprintf('d = %d\n', (minErrIndexTr - 1)); % Optimal d for lambda = 0
fprintf('d = %d\n', (minErrIndexTe - 1)); % Optimal d for lambda = 0

figure;
plot([transpose(train_errors_4a) transpose(test_errors_4a)]);
legend('Training Error','Testing Error');

% optimal regularization
train_errors_4b = zeros(length(7), 1);
test_errors_4b = zeros(length(7), 1);

lambda = lambda_optimal; % from part 2
for d = 0 : 6
    x_train_n = mapNonLinear(x_train, d);
    x_test_n = mapNonLinear(x_test, d);
    w = learnRidgeRegression(x_train_n, y_train, lambda);
    % fill code here for prediction and computing errors
    train_errors_4b(d+1) = sqrt(transpose(y_train - x_train_n * w) * ...
                                (y_train - x_train_n * w));
    test_errors_4b(d+1) = sqrt(transpose(y_test - x_test_n * w) * ... 
                                (y_test - x_test_n * w));
end

[~, minErrIndexTr] = min(train_errors_4b);
[~, minErrIndexTe] = min(test_errors_4b);
fprintf('Lambda = %f\n', lambda_optimal);
fprintf('d = %d\n', (minErrIndexTr - 1)); % Optimal d for lambda-optimal
fprintf('d = %d\n', (minErrIndexTe - 1)); % Optimal d for lambda-optimal

figure;
plot([transpose(train_errors_4b) transpose(test_errors_4b)]);
legend('Training Error','Testing Error');
toc;