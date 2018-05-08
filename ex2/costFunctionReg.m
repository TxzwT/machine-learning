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

h = 1./(e.^(-(X*theta))+1);
J = -(y'*log(h)+(1-y)'*log(1-h))/length(X);
for i=1:size(X)(2)
  if i~=1
    J+= lambda*theta(i)*theta(i)/(2*length(X));
  end
end  
#grad = X'*(1./(e.^(-(X*theta))+1)-y)/(length(X))+lambda*theta/length(X);
#grad(1) = grad(1)-lambda*theta(1)/length(X);

for i=1:length(X)
  for j=1:length(grad)
    grad(j) += (1/(1+e^-(X(i,:)*theta))-y(i))*X(i,:)(j);
  end
end
grad = grad/length(X);
for i=1:length(grad)
  if i~=1
    grad(i) += lambda*theta(i)/length(X);
  end
end
% =============================================================

end
