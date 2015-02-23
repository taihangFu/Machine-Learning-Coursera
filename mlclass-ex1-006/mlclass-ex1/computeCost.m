function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%size(hypothesis ) = 97 x 1
%hypothesis = X * theta;
%J = 1/(2*m) * sum((hypothesis - y).^2);

%================ new version

%----hypothesis aim: 97 * 1

%----- 1 x 2 * 2 x 97 = 1 x 97
hypothesis = theta' * X';

J = 1/(2*m) * sum((hypothesis' - y).^2);




% =========================================================================
end


