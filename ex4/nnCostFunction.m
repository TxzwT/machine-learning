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
ys = zeros(m,num_labels);
for i = 1:m
  ys(i,y(i)) = 1;
end
#bias
X = [ones(m,1),X];
z1 = X*Theta1';
a1 = sigmoid(z1);

X2 = [ones(m,1),a1];
z2 = X2*Theta2';
a2 = sigmoid(z2);

for i = 1:m
  s = 0;
  for j=1:num_labels
    s += ((-ys(i,j)*log(a2(i,j))-(1-ys(i,j))*log(1-a2(i,j))));
  end
  J += s;
end
%J += lambda * (sum(sum(Theta1 .^ 2))+sum(sum(Theta2 .^ 2)))/2;

for i=1:size(Theta1,1)
  for j=1:size(Theta1,2)
    if j != 1
      J += lambda * (Theta1(i,j) ^ 2)/2;
    end
  end
end
for i=1:size(Theta2,1)
  for j=1:size(Theta2,2)
    if j != 1
      J += lambda * (Theta2(i,j) ^ 2)/2;
    end
  end
end
J /= m;
%tic
for i=1:m
  #delta3 = -(ys(i,:)-a2(i,:)) .* (a2(i,:) .* (1 - a2(i,:)));
  delta3 = -(ys(i,:)-a2(i,:));
  delta2 = delta3*Theta2 .* ((X2(i,:) .* (1 - X2(i,:))));
  Theta1_grad += delta2(2:size(delta2,2))' * X(i,:);
  Theta2_grad += delta3' * X2(i,:);
end
%toc
for i=1:size(Theta1_grad,1)
  for j=1:size(Theta1_grad,2)
    if j != 1
      Theta1_grad(i,j) += lambda * Theta1(i,j);
    end
  end
end
for i=1:size(Theta2_grad,1)
  for j=1:size(Theta2_grad,2)
    if j != 1
      Theta2_grad(i,j) += lambda * Theta2(i,j);
    end
  end
end
Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end