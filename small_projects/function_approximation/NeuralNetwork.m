clear ; close all; clc





%data%
X=zeros(1000,1);
for num=1:1000
    X(num)=num/500-1;


end

Y=(sin(pi*X/200)+X.^2+X.^3)/2;


hold off;
plot(X, Y, 'rx', 'MarkerSize', 5);
X=[X,X.^2,X.^3];
fprintf("plot drawn\n");
pause;
%NN parameters%
input_layer_size = 3;
hidden_layer_size=51;
lables=1;


m = size(X, 1);

Theta1=rand(hidden_layer_size,input_layer_size+1);
Theta2=rand(lables,hidden_layer_size+1);
nn_params = [Theta1(:) ; Theta2(:)];
fprintf("Initialization COmpleted\n");
pause;
%forward Feeding of NN%


fprintf('calculating J\n')

%regularization%
lambda = 1;
J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size, lables,X, Y, lambda)
pause;

%inital parameters



fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size,lables);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
J = nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size, lables,X, Y, lambda)

%check if nn is fine%
%checkNNGradients;

lambda = 2;
%checkNNGradients(lambda);

%NN training
fprintf('\nTraining Neural Network... \n')
iter=4000;

options = optimset('MaxIter', iter);
lambda = 1;


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   lables, X, Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 lables, (hidden_layer_size + 1));
pause;

hold on;
X1=zeros(1000,1);
for iter=1:1000
    X1(iter)=iter/500-1;
    end
Y1=(sin(pi*X1/200)+X1.^2+X1.^3)/2;

plot(X1,Y1,'rx','MarkerSize',4);
plot(X1,pred([X1,X1.^2,X1.^3],Theta1,Theta2),'bo','MarkerSize',4)