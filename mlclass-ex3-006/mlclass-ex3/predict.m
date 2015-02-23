function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% 1 x 400?
%size(X)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% 25 x 401
%size(Theta1)

% 10 x 26
%size(Theta2)


%add 1 to the X 
% 5000 x 401? 
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

a_L2 = a_L2';

%-------- 3rd layer

% 10 x 26 * 26 x 5000
z_L3 = Theta2 * a_L2;

% 10 x 5000
a_L3 = sigmoid(z_L3);

%---- final
H = a_L3';


[hypothesis, h_index] = max(H, [], 2);

p = h_index;


% =========================================================================


end
