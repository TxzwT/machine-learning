function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

%for i=1:length(X)
%  tmp = -log(1+e^(X(i,:)*theta));
%  J += -tmp;
%end
h = 1./(e.^(-(X*theta))+1);
J = -(y'*log(h)+(1-y)'*log(1-h))/length(X);
grad = X'*(1./(e.^(-(X*theta))+1)-y)/(length(X));

%for i=1:length(X)
%  for j=1:length(grad)
%    grad(j) += (1/(1+e^-(X(i,:)*theta))-y(i))*X(i,:)(j);
%  end
%end
%grad = grad/length(X);





% =============================================================

end
