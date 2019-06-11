function J = computeCost(X, y, theta)

%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y


% Initialize some useful values
m = length(y); % number of training examples


% You need to return the following variables correctly 
J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



% Formula => J = 1/(2*m) Σ(H-theta(X) - y)^2
% H-theta(x) = theta0 + theta1*X   => X*theta



predictions = X*theta;              % hypothesis
sqrErrors = (predictions-y).^2;     % computing squared errors
J = 1/(2*m) * sum(sqrErrors);       % computing cost function



% =========================================================================

end
