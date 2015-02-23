function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%---------------------------DEBUGGING
%Theta1

%Theta2

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Part 1: J
X = [ones(m,1), X];

% 401 * 5000
a_L1 = X';

%--------2rd layer
% 25 x 401 * 401 * 5000
z_L2 = Theta1 * a_L1;

% 25 x 5000
a_L2 = sigmoid(z_L2);

% 5000 x 26 = 5000 x 1 + 5000 x 25
a_L2 = [ones(m,1), a_L2'];

%26 x 5000
a_L2 = a_L2';

%-------- 3rd layer

% 10 x 26 * 26 x 5000
z_L3 = Theta2 * a_L2;

% 10 x 5000
a_L3 = sigmoid(z_L3);

% 5000 x 10
hypothesis = a_L3';

%----------------------------

z = hypothesis;

%---DEBUGGING z---------SHOULD BE CORRECT!- 
%z

%size(y_matrix) = m x num_labels
y_matrix = zeros(m, num_labels);


for i = 1:m,
	y_matrix(i, y(i)) = 1;
end;

%---DEBUGGING y_matrix---------SHOULD BE CORRECT!- 
%y_matrix


%----------DEBUGGING Y1, Y0
% m x num_labels .* m x num_labels 
Y1 = y_matrix .* log(z); 

% m x num_labels .* m x num_labels
Y0 = (1 - y_matrix) .* log(1 - (z));

Y = Y1 + Y0;

%Y1

%Y0

%Y
%----------DEBUGGING 
% m x num_labels
Y = sum(Y, 1);

%
J = -1/m * sum(Y, 2);

%----Regularized
%Theta1
T1 = Theta1(:, 2:size(Theta1, 2));
%sum all elements in the T1 matrix, sum row first then coloum
T1 = sum(sum(T1.^2));


%Theta2
T2 = Theta2(:, 2:size(Theta2, 2));
T2 = sum(sum(T2.^2));

%T
Regularized = lambda/(2*m)*(T1 + T2);
 

J = -1/m * sum(Y, 2) + Regularized;


%------part 2 backpropagation---------------------


% 5000 x 10
a_L3 = a_L3';


% 5000 x 10
error_3 = a_L3 - y_matrix;

          % 25 x 10 * 10 x 5000
error_2 = Theta2(:, 2:end)' * error_3' .* sigmoidGradient(z_L2);

%--------------------------

        %10 x 5000 * 5000 x 26
        %?????????????????????????????????????????????
delta2 = error_3' * a_L2';


    % 25 x 5000 * 5000 x 401
    %?????????????????????????????????????????????
delta1 = error_2 * a_L1';


Theta2_grad = (1/m) * delta2;

Theta1_grad = (1/m) * delta1;

%----------Part 3 Regularized gradients of Theta: 
%-----------Method 2: 9a) Calculate the regularization for the entire theta gradient, 
%-----------then overwrite the (:,1) value with 0 before 9b) adding to the entire matrix.

%DEBUGGING
%Theta1
%Theta2

%M1
%10 x 26
Regularized2 = (lambda/m) * Theta2(:,2:end);

%25 x 401
Regularized1 = (lambda/m) * Theta1(:,2:end);


                % 10 x 26
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Regularized2;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Regularized1;

%-----------M2
%Regularized2(1,:) = 0;
%???????????????????????
%Regularized1(1,:) = 0;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
