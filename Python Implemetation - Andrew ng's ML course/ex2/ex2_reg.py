import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def plotData(X, y):

    pos_mask = y == 1
    neg_mask = y == 0

    plt.scatter(X[pos_mask][0].values, X[pos_mask][1].values, marker='+', label='y=1')
    plt.scatter(X[neg_mask][0].values, X[neg_mask][1].values, marker='o', label='y=0')

    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.legend()
    plt.show()


def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]

    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))

    return out


def sigmoid(z):
    g = np.zeros(z.shape, dtype=float)
    
    g = 1 / (1 + np.exp(-z))
    return g


def costFunctionReg(theta, X, y, _lambda):
    
    # Initialize some useful values
    m = len(y)  # number of training examples
    J = 0

    # calling sigmoid function
    h_thetaX = sigmoid(X.dot(theta))

    # cost function
    J = - 1 * (1 / m) * ((y.T.dot(np.log(h_thetaX))) + (1 - y).T.dot(np.log(1 - h_thetaX))) + (_lambda / (2*m))*np.sum(np.square(theta[1:]))

    return J


def gradientReg(theta, X, y, _lambda):

    # Initialize some useful values
    m = len(y)  # number of training examples
    grad = np.zeros(initial_theta.shape)

    # calling sigmoid function
    h_thetaX = sigmoid(X.dot(theta))

    # gradient of cost function
    grad = (1 / m) * X.T.dot(h_thetaX - y)
    grad[1:] = grad[1:] + (_lambda / m)*theta[1:]

    return grad.flatten()


#######################################################
# Part 0. Initialization
#######################################################

print('\nPlotting Data ...\n')

data = pd.read_csv('ex2data2.txt', header = None)
X = data.iloc[:,0:2]
y = data.iloc[:,2]

plotData(X, y)


#######################################################
# Part 1. Regularized Logistic Regression
#######################################################

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X.iloc[:,0], X.iloc[:,1])
[m, n] = X.shape

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1
_lambda = 1

y = y[:,np.newaxis]

# Compute and display initial cost and gradient for regularized logistic regression
cost = costFunctionReg(initial_theta, X, y, _lambda)
grad = gradientReg(initial_theta, X, y, _lambda)

print('Cost at initial theta (zeros): ', cost[0][0])
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only: ', grad[0:5])
print('Expected gradients (approx) - first five values only: [ 0.0085 0.0188 0.0001 0.0503 0.0115 ]\n')

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones((n, 1))
cost = costFunctionReg(test_theta, X, y, 10)
grad = gradientReg(test_theta, X, y, 10)

print('Cost at test theta (ones and with lambda = 10): ', cost[0][0])
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only: ', grad[0:5])
print('Expected gradients (approx) - first five values only: [ 0.3460 0.1614 0.1948 0.2269 0.0922 ]\n')


#######################################################
# Part 2. Accuracy
#######################################################

# calling fmin_tnc function of scipy
temp = opt.fmin_tnc(func = costFunctionReg, 
                    x0 = initial_theta.flatten(),fprime = gradientReg, 
                    args = (X, y.flatten(), _lambda), disp=0)

# getting optimised theta
theta_optimized = temp[0]

# prediction
pred = np.round(sigmoid(X.dot(theta_optimized)))
y = y.flatten()
print('Train Accuracy: ', np.mean(pred == y) * 100)
print('Expected accuracy approx (with lambda = 1): 83.1\n')
