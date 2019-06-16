import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def sigmoid(z):
    g = np.zeros(z.shape, dtype=float)
    
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoidGradient(z):
    return(sigmoid(z)*(1-sigmoid(z)))


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
    Theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    Theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,(hidden_layer_size+1))

    # Setup some useful variables
    m = X.shape[0]
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # Part 1: Feedforward the neural network and return the cost in the variable J. After implementing Part 1, you can verify that your
    # cost function computation is correct by verifying the cost computed in ex4.m

    ones = np.ones((m,1))
    X = np.hstack((ones, X))

    a2 = sigmoid(X.dot(Theta1.T))

    a2 = np.hstack((ones, a2))
    h_theta = sigmoid(Theta2.dot(a2.T))

    y_new = np.zeros((num_labels, m)).T
    
    for i in range(0, m):
        y_new[i][y[i][0] - 1] = 1

    J = -1*(1/m)*np.sum((np.log(h_theta.T)*(y_new)+np.log(1-h_theta).T*(1-y_new)))

    Reg = (_lambda/(2*m))*(np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])))

    J = J + Reg
    
    return J
    


#######################################################
# Part 1. Loading and Displaying Data
#######################################################

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']

fig, axarr = plt.subplots(5,5,figsize=(5,5))

for i in range(5):
    for j in range(5):
       axarr[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((20,20), order = 'F'))          
       axarr[i,j].axis('off')

#plt.show()

# Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25
output_layer_size = 10
num_labels = 10    # 10 labels, from 1 to 10 ("0" is mapped to 10)


#######################################################
# Part 2. Loading Parameters
#######################################################

# Load the weights into variables Theta1 and Theta2
weightData = loadmat('ex4weights.mat')
Theta1 = weightData['Theta1']
Theta2 = weightData['Theta2']

# Unroll parameters 
nn_params = np.r_[Theta1.ravel(), Theta2.ravel()]


#######################################################
# Part 3. Compute Cost (Feedforward)
#######################################################

print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
_lambda = 0

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print('Cost at parameters (loaded from ex4weights) (this value should be about 0.287629)\n', J)


#######################################################
# Part 4. Implement Regularization
#######################################################

print('\n\nChecking Cost Function (w/ Regularization)\n')

_lambda = 1

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print('Cost at parameters (loaded from ex4weights) (this value should be about 0.383770)\n', J)


#######################################################
# Part 5. Sigmoid Gradient
#######################################################
 
print('\nEvaluating sigmoid gradient...\n')
g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]).reshape(-1, 1))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]: ', g.ravel())