clear all;

rand('state',0);
randn('state',0);

addpath('Libary');
%%%%%%%%%%%%% Generate the two moon data set
N = 1000; % Number of data points 
d = 20; % d is the input dimension;
[X,Y]=GD_GenerateData(2,N,d);
P = cvpartition(Y,'Holdout',0.9);
TrainX = X(:, P.training); TrainY = Y(P.training); 

%%%%%%%%%%%%% Computing FTSD solution

% Use 1-NN Classifier based on Euclidean distance
NumNeighbors = 1; Metric = 'euclidean';

K = 7; gamma = 1; alpha = 1e-5; k = 6; M = 10;  % Use default parameter setting
[T_FTSD ~] = MPDA(TrainX, TrainY, K, gamma, alpha, k, M);
Z = X'*T_FTSD;
Class = knnclassify(Z(P.test,:), Z(P.training,:), Y(P.training), NumNeighbors, Metric);
LossSTSD = sum(Y(P.test)~= Class)/P.TestSize