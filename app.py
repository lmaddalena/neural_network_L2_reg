import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from neuralnetwork import *
from utils import *


def main():
    print("Deep Neural Network with L2 regularization")
    print("Rif.: Improving Deep Neural Networks - week 1 (revisited)")

    print("")
    print("Loading dataset...\n")
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    # Explore your dataset 
    m_train = train_X.shape[1]
    num_px = train_X.shape[0]
    m_test = test_X.shape[1]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("train_X shape: " + str(train_X.shape))
    print ("train_Y shape: " + str(train_Y.shape))
    print ("test_X shape: " + str(test_X.shape))
    print ("test_Y shape: " + str(test_Y.shape))
    print ("-----------------------------------------------")
    
    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    plt.show()

    layers_dims = [num_px, 20, 3, 1]
    parameters = model(train_X, train_Y, layers_dims=layers_dims, lambd = 0.7, learning_rate=0.3, num_iterations=30000, print_cost=True)
    predictions_train = predict(train_X, parameters)
    print ("On the train set accuracy: %f" %(np.mean(predictions_train == train_Y)))

    predictions_test = predict(test_X, parameters)
    print ("On the test set accuracy: %f" %(np.mean(predictions_test == test_Y)))



if __name__ == "__main__":
    main()