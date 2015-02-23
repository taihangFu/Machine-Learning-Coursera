function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% m x size(theta) * size(theta) x 1
hypothsis = X * theta;
%
s = (hypothsis - y).^2;

J = 1/(2*m) * sum(s);

r_s = theta.^2;

r_s(1) = 0;

J = J + lambda/(2*m) * sum(r_s);

%size(J)
%-------grad

%r = lambda/m * theta;
%r(1) = 0;

%grad = 1/m * X'*(hypothsis - y) + r;

%-------theta0------------

% 1 * 1 = 1 x m * m x 1
grad0 = 1/m * X(:, 1)' * (hypothsis - y); 


%------------theta 1 to m ----------------

% 27 x 1
regularized1 = lambda/m * theta(2:end, :);

% 27 x 1 = 27 x m * m x 1
grad1 = (1/m * X(:,2:end)' * (hypothsis - y)) + regularized1; 

%----------------
grad(1) = grad0;

grad(2:end) = grad1;









% =========================================================================

grad = grad(:);

end
