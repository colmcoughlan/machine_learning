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


s = max(size(theta));
h=(X*theta);
temp = h - y;

J = ( ((temp'*temp) + lambda*(theta(2:s)'*theta(2:s))) / (2.0 *m));


grad_reg = theta;
grad_reg = grad_reg * (lambda/m);
grad_reg(1) = 0;

grad = X'*(temp)/m + grad_reg;



% =========================================================================

grad = grad(:);

end
