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

% 这样就能把 Y 分成 [m,10] ，其中每行为 1 的对应 列 为本来的要预测的数值
% 这样就把分类算法的分类最后的 output 的 Y 预测出来了
I = eye(num_labels) ; 
Y = zeros(m , num_labels) ; 
for i = 1 : m 
    Y(i , : ) = I(y(i) , :) ;
end  

% feedforward 
a1 = [ones(m , 1) , X ] ; % 第一层 5000 * 401
z2 = a1 * Theta1' ; % [5000   401] * [25   401]' = [5000 25]  
a2 = [ones(m , 1) , sigmoid(z2)] ; % 第二层 5000 *26 
z3 = a2 * Theta2' ; % [5000 26] * [10 26]' = [5000 10] 
h = sigmoid(z3) ; 

% calcute penalty 其实这个函数就是 Y(i.j) .* h(i,j) 的矩阵，然后把所有的值求和
for i = 1 : m 
    first_term = -Y(i , :) .* log(h(i,:)) ; 
    second_term = (1 - Y(i , :)) .* log(1 - h(i , :)) ; 
    J += sum(first_term - second_term) ; 
end;
J = J / m ;

% regulariztion 
t1 = Theta1(: , 2:size(Theta1,2)) ; % 把 第一列 拿掉
t2 = Theta2(: , 2:size(Theta2,2)) ; % 把 第一列 拿掉
% 所有 Theta 的平方和
Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);
J= J + Reg ; 


for i = 1 : m 
    hi = h(i , :) ; %[1 10]
    yi = Y(i , :) ; %[1 10]  
    d3t = hi - yi ;  %[1 10] 第一次偏导
    
    Theta2_grad = Theta2_grad + d3t' * a2(i,:) ; % [10 26]
   
    z2t = [ones(1,1) , z2(i , :)] ;  % [1 26] 还没有求 sigmod 的那一层
    
    d2t = ((Theta2)' * d3t')' .* sigmoidGradient(z2t) ; %这个是 back 算法,大佬的算法 [1 26]
    d2t = d2t(2:end) ;  %[1 25]
    Theta1_grad = Theta1_grad + d2t' * a1(i,:) ; 
end ; 

%Theta1_grad = Theta1_grad ./ m ; 
%Theta2_grad = Theta2_grad ./ m ; 




% regulariztion 
Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));

Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
