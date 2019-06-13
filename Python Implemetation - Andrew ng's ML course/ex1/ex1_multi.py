import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt

def featureNormalize(X):
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    m = len(y) # number of training examples
    J = 0
    h = X.dot(theta)
    J = 1/(2*m)*np.sum(np.square(h-y))
    return J


def normalEqn(X, y):
    theta = np.zeros([3, 1])
    theta = np.linalg.inv(X.T.dot(X))
    theta = theta.dot(X.T)
    theta = theta.dot(y)
    return theta


def gradientDescentMulti(X, y, theta, alpha, iterations):

    m = len(y) # number of training examples
    J_history = np.zeros(iterations)
    
    for i in range(iterations):

        error = np.dot(X, theta) - y
        temp = np.dot(X.T, error)
        theta = theta - (alpha/m) * temp

        J_history[i] = computeCostMulti(X, y, theta)

    return theta, J_history


#######################################################
# Part 1. Feature Normalization
#######################################################

data = pd.read_csv('ex1data2.txt', header = None)
X = data.iloc[:,0:2]
y = data.iloc[:,2]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: \n')
print(data.head(10))

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
ones = np.ones((m,1))
X = np.hstack((ones, X))


#######################################################
# Part 2. Gradient descent
#######################################################

print('\nRunning gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# converting y to higher dim
y = y[:,np.newaxis]

# Init Theta and Run Gradient Descent 
theta = np.zeros((3,1))
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
x_values = [x for x in range(1,num_iters+1)]
plt.plot(x_values, J_history, '-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: ', theta.ravel())

# Estimate the price of a 1650 sq-ft, 3 br house
price = 0

temp = np.array([1, 1650, 3], dtype=float)

# normalising
temp[1] = (temp[1] - mu[0])/sigma[0]
temp[2] = (temp[2] - mu[1])/sigma[1]

# prediction
price = theta.T.dot(temp)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ', price[0])


#######################################################
# Part 3. Normal Equations
#######################################################

data = pd.read_csv('ex1data2.txt', header = None)
X = data.iloc[:,0:2]
y = data.iloc[:,2]
m = len(y)

# Add intercept term to X
ones = np.ones((m,1))
X = np.hstack((ones, X))

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('\nTheta computed from the normal equations: ', theta.ravel())

# Estimate the price of a 1650 sq-ft, 3 br house
price = 0

temp = np.array([1, 1650, 3], dtype=float)

# prediction
price = theta.T.dot(temp)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ', price)
