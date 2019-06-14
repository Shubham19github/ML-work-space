import numpy as np
from scipy.io import loadmat


def sigmoid(z):
    g = np.zeros(z.shape, dtype=float)
    
    g = 1 / (1 + np.exp(-z))
    return g


def predict(Theta1, Theta2, X):

    [m, n] = X.shape

    ones = np.ones((m, 1))
    X = np.hstack((ones, X))

    a1 = sigmoid(X.dot(Theta1.T))
    a2 = np.hstack((ones, a1))
    h_theta = sigmoid(a2.dot(Theta2.T))

    return(np.argmax(h_theta, axis=1)+1)


#######################################################
# Part 1. Loading Data
#######################################################

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

# Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10 ("0" is mapped to 10)

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
weightData = loadmat('ex3weights.mat')
Theta1 = weightData['Theta1']
Theta2 = weightData['Theta2']

#######################################################
# Part 2: Implement Predict
#######################################################

pred = predict(Theta1, Theta2, X)
pred = [e if e else 10 for e in pred]
print('\nTraining Set Accuracy: ', np.mean(pred == y.flatten())*100)