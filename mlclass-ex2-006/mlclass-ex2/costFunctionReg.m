function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% m x 1
z = X * theta;

% m x 1 * m x 1 = m x 1
Y1 = y .* log(sigmoid(z)); 

% m x 1
Y0 = (1 - y) .* log(1 - (sigmoid(z)));

% m x 1
regularized = lambda/(2*m) * sum((theta(2:end)).^2);


%
J = -1/m * sum(Y1 + Y0) + regularized;

%-------------------grad-------------------------------------------

%-------theta0------------
% m x 1
hypothesis = sigmoid(z);

% 1 * 1 = 1 x m * m x 1
grad0 = 1/m * X(:, 1)' * (hypothesis - y); 


%------------theta 1 to m ----------------
hypothesis = sigmoid(z);

% 27 x 1
regularized1 = lambda/m * (theta(2:end, :));

% 27 x 1 = 27 x m * m x 1
grad1 = (1/m * X(:,2:end)' * (hypothesis - y)) + regularized1; 

%----------------
grad(1) = grad0;

grad(2:end) = grad1;


% =============================================================

end
