import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def plotData(X, y):

    pos_mask = y == 1
    neg_mask = y == 0

    plt.scatter(X[pos_mask][0].values, X[pos_mask][1].values, marker='+', label='Admitted')
    plt.scatter(X[neg_mask][0].values, X[neg_mask][1].values, marker='o', label='Not Admitted')

    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

    plt.legend()


def sigmoid(z):
    g = np.zeros(z.shape, dtype=float)
    
    g = 1 / (1 + np.exp(-z))
    return g


def costFunction(theta, X, y):
    
    # Initialize some useful values
    m = len(y)  # number of training examples
    J = 0

    # calling sigmoid function
    h_thetaX = sigmoid(X.dot(theta))

    # cost function
    J = - 1 * (1 / m) * ((y.T.dot(np.log(h_thetaX))) + (1 - y).T.dot(np.log(1 - h_thetaX)))

    return J


def gradient(theta, X, y):

    # Initialize some useful values
    m = len(y)  # number of training examples
    grad = np.zeros(initial_theta.shape)

    # calling sigmoid function
    h_thetaX = sigmoid(X.dot(theta))

    # gradient of cost function
    grad = (1 / m) * X.T.dot(h_thetaX - y)

    return grad.flatten()


#######################################################
# Part 1. Plotting Data
#######################################################

# The first two columns contains the exam scores and the third column contains the label.
print('\nPlotting Data ...\n')

data = pd.read_csv('ex2data1.txt', header = None)
X = data.iloc[:,0:2]
y = data.iloc[:,2]
m = len(y)

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

plotData(X, y)

#######################################################
# Part 2. Compute Cost and Gradient
#######################################################

# Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = X.shape

# Add intercept term to x and X_test
ones = np.ones((m,1))
X = np.hstack((ones, X))

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

y = y[:,np.newaxis]


# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

print('Cost at initial theta (zeros): ', cost[0][0])
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): ', grad.ravel())
print('Expected gradients (approx): [ -0.1000  -12.0092  -11.2628 ]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2]).reshape(3,1)

cost = costFunction(test_theta, X, y)
grad = gradient(test_theta, X, y)

print('Cost at test theta (zeros): ', cost[0][0])
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta (zeros): ', grad.ravel())
print('Expected gradients (approx): [ 0.043  2.566  2.647 ]\n')


#######################################################
# Part 3. Optimizing using fminunc
#######################################################

# calling fmin_tnc function of scipy
temp = opt.fmin_tnc(func = costFunction, 
                    x0 = initial_theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()), disp=0)

# getting optimised theta
theta_optimized = temp[0]

J = costFunction(theta_optimized[:,np.newaxis], X, y)

print('\nCost at theta found by fminunc: ', J[0][0])
print('Expected cost (approx): 0.203\n')
print('theta: ', theta_optimized)
print('Expected theta (approx): [ -25.161 0.206 0.201 ]\n')


# plotting decision boundary
plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0] + np.dot(theta_optimized[1],plot_x))
plt.plot(plot_x, plot_y)
plt.show()


#######################################################
# Part 4. Predict and Accuracies
#######################################################

# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2 

score_vector = np.array([1, 45, 85])
prob = sigmoid(score_vector.dot(theta_optimized))
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)
print('Expected value: 0.775 +/- 0.002\n\n')

pred = np.round(sigmoid(X.dot(theta_optimized)))
y = y.flatten()
print('Train Accuracy: ', np.mean(pred == y) * 100)
print('Expected accuracy (approx): 89.0\n')