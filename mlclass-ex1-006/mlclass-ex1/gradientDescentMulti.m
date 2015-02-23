function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


 % Method 1
    
    % ----size(hypothesis) = 97 x 1
    %hypothesis = X * theta; 


    % -----size(S) = 2 x 97 * 97 x 1 = 2 x 1 <- multiply Matrix add element up, thus no need sum() here
    %S = X' * (hypothesis - y);

    %2 x 1
    %gamma = 1/m * alpha * S; 

    %theta = theta - gamma;

%------------- new Version, works for any size!
    %1x47
    hypothesis = theta' * X'; 

    %3x1 = 3xm * mx1
    S = X' * (hypothesis'- y);

    %3x1
    gamma = 1/m * alpha * S; 

    %3x1
    theta = theta - gamma;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
