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

misui_weights = zeros(m,1);

for nisui_i = 1:m

    a1 = X(nisui_i,:)';
    a1_with_1 = [1;a1];

    z2 = Theta1 * a1_with_1;
    a2 = sigmoid(z2);
    a2_with_1 = [1;a2];

    z3 = Theta2 * a2_with_1;
    a3 = sigmoid(z3);

    y_i = zeros(num_labels,1);
    y_i(y(nisui_i)) = 1;
 
    Jvec = -y_i .* log(a3) - (1-y_i) .* log(1-a3);
    misui_weights(nisui_i) = sum(Jvec);
end

J = 1/m * sum(misui_weights); 

% ok % ok % ok % ok % ok % ok % ok % ok % ok % ok 

sum = 0;
[r1,c1] = size(Theta1);
for rj = 1:r1
    for ck = 2:c1 % don't take into account the bias (first column)
        sum = sum + Theta1(rj, ck) * Theta1(rj, ck);
    end
end

[r2,c2] = size(Theta2);
for rj = 1:r2
    for ck = 2:c2 % don't take into account the bias (first column)
        sum = sum + Theta2(rj, ck) * Theta2(rj, ck);
    end
end

J = J + (lambda / (2 * m)) * sum;


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


Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for t = 1:m % t -> nisui_i 
  
    %1 Set the input layer’s values (a(1)) to the t-th training example x(t)
   
    a1 = X(t,:)';
    a1_with_1 = [1;a1];

    z2 = Theta1 * a1_with_1; % SIZE [25 х 401] * [401 x 1] = 25 x 1
    a2 = sigmoid(z2);        % SIZE [25 x 1]
    a2_with_1 = [1;a2];      % SIZE [26 x 1]

    z3 = Theta2 * a2_with_1; % SIZE [10 х 26] * [26 x 1] = 10 x 1
    a3 = sigmoid(z3);

    %2 δ(3) = (a(3) − yk),
    
    y_t = zeros(num_labels,1);
    y_t(y(t)) = 1;           % SIZE 10 x 1
    
    
    % 3 %%%%%%%%%%%%%%%%%%%%%
    delta_3 = a3 - y_t;      % SIZE [10 x 1]
    
    % 2 %%%%%%%%%%%%%%%%%%%%%
    xxx = Theta2' * delta_3;
    delta_2 = xxx(2:end,:) .* sigmoidGradient(z2); % SIZE [25 x 10] * [10 x 1] = 25 x 1
    
    % no Delta3
    Delta2 = Delta2 + delta_3 * a2_with_1';
    Delta1 = Delta1 + delta_2 * a1_with_1';
    
end

Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

[r, c] = size(Theta1_grad);

for ri = 1:r
    for ci = 2:c
        Theta1_grad(ri, ci) = Theta1_grad(ri, ci) + (lambda / m) * Theta1(ri, ci);
    end
end


[r, c] = size(Theta2_grad);

for ri = 1:r
    for ci = 2:c
        Theta2_grad(ri, ci) = Theta2_grad(ri, ci) + (lambda / m) * Theta2(ri, ci);
    end
end















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
