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

n_features = size(theta)(1)

cost1 = 0
for i=1:m
    h_i = sigmoid(theta'*X(i,:)')
    cost1 = cost1 + (-y(i)*log(h_i) - (1-y(i))*(log(1-h_i)))
end
cost1 = (1.0/m) * cost1

cost2 = 0
for i=2:n_features % We dont regularize theta(0)
    cost2 = cost2 + (theta(i,1) ^ 2)
end
cost2 = (lambda/(2.0*m)) * cost2

J = cost1 + cost2

for i=1:n_features
    gradient = 0
    for j=1:m
       h_j = sigmoid(theta'*X(j,:)')
       gradient = gradient + (h_j - y(j,1))*X(j,i)
    end
    if (i == 1)
       grad(i,1) = (1.0/m) * gradient
    else
       grad(i,1) = (1.0/m) * gradient + (lambda*1.0/m)*theta(i,1)
    endif
end


% =============================================================

end
