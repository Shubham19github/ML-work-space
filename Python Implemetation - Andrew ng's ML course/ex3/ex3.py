import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    g = np.zeros(z.shape, dtype=float)
    
    g = 1 / (1 + np.exp(-z))
    return g


def lrcostFunctionReg(theta, X, y, _lambda):
    
    # Initialize some useful values
    m = len(y)  # number of training examples
    J = 0

    # calling sigmoid function
    h_thetaX = sigmoid(X.dot(theta))

    # cost function
    J = - 1 * (1 / m) * ((y.T.dot(np.log(h_thetaX))) + (1 - y).T.dot(np.log(1 - h_thetaX))) + (_lambda / (2*m))*np.sum(np.square(theta[1:]))

    return J


def lrgradientReg(theta, X, y, _lambda):

    # Initialize some useful values
    m = len(y)  # number of training examples
    grad = np.zeros(theta.shape)

    # calling sigmoid function
    h_thetaX = sigmoid(X.dot(theta))

    # gradient of cost function
    grad = (1 / m) * X.T.dot(h_thetaX - y)
    grad[1:] = grad[1:] + (_lambda / m)*theta[1:]

    return grad.flatten()


def oneVsAll(X, y, num_labels, _lambda):

    [m, n] = X.shape
    all_theta = np.zeros((num_labels, n))
    initial_theta = np.zeros((n, 1))

    for i in range(num_labels):
        digit_class = i if i else 10
        all_theta[i] = opt.fmin_cg(f = lrcostFunctionReg, x0 = initial_theta.flatten(),  
                                    fprime = lrgradientReg, 
                                    args = (X, (y == digit_class).flatten(), _lambda), 
                                    maxiter = 50)

    return(all_theta)


def predictOneVsAll(all_theta, X):
    prob = sigmoid(X.dot(all_theta.T))
    return(np.argmax(prob, axis=1))


#######################################################
# Part 1. Loading and Displaying Data
#######################################################

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

fig, axarr = plt.subplots(5,5,figsize=(5,5))

for i in range(5):
    for j in range(5):
       axarr[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((20,20), order = 'F'))          
       axarr[i,j].axis('off')

plt.show()

# Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10 ("0" is mapped to 10)


#######################################################
# Part 2. Vectorize Logistic Regression
#######################################################

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2]).reshape(-1, 1)

temp = np.arange(1.0, 16.0).reshape(3, 5)/10
ones = np.ones((5, 1))
X_t = np.hstack((ones, temp.T))
y_t = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
lambda_t = 3

J = lrcostFunctionReg(theta_t, X_t, y_t, lambda_t)
grad = lrgradientReg(theta_t, X_t, y_t, lambda_t)

print('\nCost: ', J[0][0])
print('Expected cost: 2.534819')
print('Gradients: ', grad)
print('Expected gradients: [0.146561 -0.548558 0.724722 1.398003]\n')


#######################################################
# Part 3. One-vs-All Training
#######################################################

print('\nTraining One-vs-All Logistic Regression...\n')

_lambda = 0.1
[m, n] = X.shape
ones = np.ones((m,1))
X = np.hstack((ones, X))

all_theta = oneVsAll(X, y, num_labels, _lambda)

#######################################################
# Part 4. Predict for One-Vs-All
#######################################################

pred = predictOneVsAll(all_theta, X)
pred = [e if e else 10 for e in pred]
print('\nTraining Set Accuracy: ', np.mean(pred == y.flatten())*100)