import numpy as np
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt

def plotData(X, y):
    axes = plt.gca()
    axes.set_xlim([4,25])
    axes.set_ylim([-5,25])
    plt.scatter(X, y, color='r', marker='*', label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')


def computeCost(X, y, theta):
    m = len(y) # number of training examples
    J = 0
    h = X.dot(theta)
    J = 1/(2*m)*np.sum(np.square(h-y))
    return J


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y) # number of training examples
    J_history = np.zeros(iterations)
    
    for i in range(iterations):

        error = np.dot(X, theta) - y
        temp = np.dot(X.T, error)
        theta = theta - (alpha/m) * temp

        J_history[i] = computeCost(X, y, theta)

    return theta


#######################################################
# Part 1. Warmup excercise
#######################################################

# printing Identity matrix
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
A = np.identity(5)
print(A)

#######################################################
# Part 2. Plotting Data
#######################################################

print('\nPlotting Data ...\n')

data = pd.read_csv('ex1data1.txt', header = None)
X = data.iloc[:,0]
y = data.iloc[:,1]
m = len(y)

plotData(X, y)


#######################################################
# Part 3. Cost and Gradient descent
#######################################################

# Add a column of ones to x
X = X[:,np.newaxis]
y = y[:,np.newaxis]

ones = np.ones((m,1))
X = np.hstack((ones, X))

# initialize fitting parameters
theta = np.zeros([2,1])

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

# compute and display initial cost
J = computeCost(X, y, theta)

print('With theta = [0 ; 0]\nCost computed = %f' %J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
theta = np.array([-1, 2]).reshape(2,1)
J = computeCost(X, y, theta)

print('\nWith theta = [-1 ; 2]\nCost computed = %f' %J)
print('Expected cost value (approx) 54.24\n')


print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:', theta.ravel())
print()
print('Expected theta values (approx) [-3.6303  1.1664]\n')

# Predict values for population sizes of 35,000 and 70,000

predict1 = theta.T.dot([1, 3.5])*10000
print('For population = 35,000, we predict a profit of ', predict1)

predict2 = theta.T.dot([1, 7])*10000
print('For population = 70,000, we predict a profit of ', predict2)

# Plot the linear fit
plt.plot(X[:,1], np.dot(X, theta), '-', label='Linear regression')

plt.legend()
plt.show()