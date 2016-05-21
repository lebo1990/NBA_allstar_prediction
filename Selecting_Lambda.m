%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 23;  % 20x20 Input Images of Digits
hidden_layer_size = 46;   % 25 hidden units
num_labels = 2;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%
% 
% % Load Training Data
% fprintf('Loading and Visualizing Data ...\n')
% 
% load('ex4data1.mat');
% m = size(X, 1);
% 
% % Randomly select 100 data points to display
% sel = randperm(size(X, 1));
% sel = sel(1:100);
% 
% displayData(X(sel, :));
% 
% fprintf('Program paused. Press enter to continue.\n');
% % pause;

load('allstar_train_X.mat');
load('allstar_train_y.mat');
load('TestX.mat');
load('Testy.mat');
m = size(X, 1);

meanX = mean(X, 1);
stdX = std(X, 1);

for i = 1: size(X, 1)
    X(i, :) = (X(i, :) - meanX) ./ stdX;
end

for i = 1: size(TestX, 1)
    TestX(i, :) = (TestX(i, :) - meanX) ./ stdX;
end

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%

% randomly choose training set and cross validation set
Xy = [X, y; TestX, Testy];

ratio = 0.6;
num = ceil(size(Xy, 1) * ratio);

%  value to see how more training helps.
options = optimset('MaxIter', 50);

% Lambda
lambda_vec = [0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
pred_train = zeros(length(lambda_vec), 1);
pred_val = zeros(length(lambda_vec), 1);

for i = 1: length(lambda_vec)
    fprintf('\nTraining Neural Network... \n')
    
    randidx = randperm(size(Xy, 1)); %shuffle
    Xtrain = Xy(randidx(1:num), 1: end - 1);
    ytrain = Xy(randidx(1:num), end);
    
    Xval = Xy(randidx(num + 1: end), 1: end - 1);
    yval = Xy(randidx(num + 1: end), end);
    
%  Initilization   
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
% Create "short hand" for the cost function to be minimized
    costFunction_train = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, ytrain, lambda_vec(i));
                                                         
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
    [nn_params_train, ~] = fmincg(costFunction_train, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params_train(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params_train((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
    
    nn_params = [Theta1(:) ; Theta2(:)];

	error_train(i) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   Xtrain, ytrain, 0);
	% error_train(i) = linearRegCostFunction(X, y, theta, lambda_vec(i));
	error_val(i) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   Xval, yval, 0);
	% error_val(i) = linearRegCostFunction(Xval, yval, theta, lambda_vec(i));

    pred_train(i) = mean(double(predict(Theta1, Theta2, Xtrain) == ytrain)) * 100;
    pred_val(i) = mean(double(predict(Theta1, Theta2, Xval) == yval)) * 100;
    
end

% close all;
figure();
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('Lambda \t Traing set prediction \t cross validation set prediction\n');
for i = 1: length(lambda_vec)
    fprintf('%.3f \t', lambda_vec(i));
    fprintf('\t %.3f \t', pred_train(i));
    fprintf('\t \t \t \t %f \t \n', pred_val(i));
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

    
