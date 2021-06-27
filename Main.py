"""
Aaron Safer-Rosenthal | 17asr | 20068164

Generates training and testing data, and runs the various models with it.
All models use backpropogation, gradient descent and the sigmoidal activation
function. I have coded model 1 and implement models 2a/2b. Models 1 and 2a use
the same parameter values, and model 2b uses different ones.
"""

import numpy as np
import tensorflow as tf
from Model import Network
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#flags from the beginning of when I started the program
def flags1(trx, ntry, tx, ty):
    print(trx)
    print(ntry)
    print(tx)
    print(ty)

#flags from the beginning of when I started the program
def flags2(trx, ntry, tx, ty):
    print(len(trx))
    print(len(ntry))
    print(len(tx))
    print(len(ty))

#My backpropogation network
def model1(numInputLayer, numHiddenNodes, numOutputNodes, numEpochs,
           miniBatchSize, learningRate):

    #importing training and testing data
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(
        r"C:\Users\asafe\Documents\Work\CISC 452\Assignment 2\mnist.npz")

    #flags1(train_X, train_y, test_X, test_y) #flag(s)
    #flags2(train_X, train_y, test_X, test_y) #flag(s)

    #formatting the training and testing data
    training_inputs = [np.reshape(x, (784, 1)) for x in train_X]
    training_results = [vectorize(y) for y in train_y]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in test_X]
    test_data = zip(test_inputs, test_y)

    network = Network([numInputLayer, numHiddenNodes, numOutputNodes])
    confusionMatrix = network.gradientDescent(training_data, numEpochs,
                                          miniBatchSize, learningRate,
                                          test_data)
    """
    for some reason when I try to make the confusion matrix seperately
    by just using the line
        confusionMatrix = net.makeConfusionMatrix(test_data)
    it does not work. However, when making it straight from the gradient descent
    funciton, it does work. I have no explanation for it and tried to
    troubleshoot for a long time. I hope this is not an issue.
    """
    
    #printing results
    model1Results("training", confusionMatrix[0], network)
    model1Results("testing", confusionMatrix[1], network)


#Model that is imported from a predefined library (Scikit-learn)
#Uses the same parameters as model 1.
#(Is still a backpropogation network using gradient descent and
#sigmoidal activation function
def model2A(numInputLayer, numHiddenNodes, numOutputNodes, numEpochs,
           miniBatchSize, learningRate):

    #Getting training and testing data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X /= 255 #normalizing data to be between 0-1
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=60000, test_size=10000, shuffle=False)

    mlp = MLPClassifier(hidden_layer_sizes=(numHiddenNodes,),
                        max_iter=numEpochs, alpha=1e-4,
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=learningRate)

    #ignoring the convergence warning, since it is thrown when I traintest with
    #a low number of epochs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        mlp.fit(X_train, y_train)

    #print("Training set score: %f" % mlp.score(X_train, y_train)) #results
    #print("Test set score: %f" % mlp.score(X_test, y_test)) #results

    model2Results(mlp, X_train, y_train, "Training")
    model2Results(mlp, X_test, y_test, "Testing")

#Model that implements the same model as 2a, but with different parameters
def model2B():

    #network parameters, change as desired
    numInputLayer = 784
    numHiddenNodes = 200
    numOutputNodes = 10
    numEpochs = 100
    miniBatchSize = 10
    learningRate = 0.05

    #Getting training and testing data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X /= 255 #normalizing data to be between 0-1
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=60000, test_size=10000, shuffle=False)

    mlp = MLPClassifier(hidden_layer_sizes=(numHiddenNodes,),
                        max_iter=numEpochs, alpha=1e-4,
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=learningRate)

    #ignoring the convergence warning, since it is thrown when I train/test with
    #a low number of epochs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        mlp.fit(X_train, y_train)

    #printing results
    model2Results(mlp, X_train, y_train, "Training")
    model2Results(mlp, X_test, y_test, "Testing")

#Input: Type of either training or testing, and a confusion matrix
#Output, printed confusion matrix, precision and recall
def model1Results(t, matrix, network): 
    print(t + ":") #for readability
    printMatrix(matrix) #flag
    precision = network.getPrecision(matrix)
    Recall = network.getRecall(matrix)
    print("Precision: " + str(precision*100) + "%")
    print("Recall: " + str(Recall*100) + "%")

#Input: Trained network, X data (training or testing) and y data (same)
#Output, printed confusion matrix, precision and recall
def model2Results(mlp, X_set, y_set, t):

    predictions = mlp.predict(X_set)
    #print("Predictions:", predictions) #flag
    #print("y_test", y_test) #flag
    print(t + " results:") #for readability
    precision = precision_score(y_set, predictions, average='micro')
    recall = recall_score(y_set, predictions, average='micro')
    print("Precision: " + str(precision*100) + "%")
    print("Recall: " + str(recall*100) + "%")
    confusionMatrix = makeConfusionMatrix(predictions, y_set)
    printMatrix(confusionMatrix)

#Input: Test data
#Output: Confusion matrix (as a 2D-list)
def makeConfusionMatrix(predictions, targets):

    matrix = matrix = [[0 for _ in range(10)] for _ in range(10)]
    fullResults = list(zip(predictions, targets))
    #print(fullResults) #flag
    for (prediction, target) in fullResults:
        """output x-axis, target y-axis
        i.e. row [5,1,2,0] means that it correctly identified 5 1s as 1,
        misidentified 1 2 as 1, 2 3s as 1 and no 4s as 1.
        """
        #print(x,y) #flag
        matrix[int(prediction)][int(target)] += 1
    #print("matrix in function: ", matrix) #flag
    return matrix

#Input: Confusion Matrix
#Output: Prints a better formatted confusion matrix
def printMatrix(matrix):

    print("Confusion matrix:") #for readability
    for i in range(len(matrix)):
        print(matrix[i])

#Input: Sample value from training or testing data
#Output: 10-d vector, with 1.0 in position i and 0s in all other potitions
def vectorize(i):

    result = np.zeros((10, 1))
    result[i] = 1.0
    return result

#Main function which drives the program
#Sets parameter values for models 1 and 2a
#Runs the model functions
def main():

    #Setting parameter values for models 1 and 2a
    numInputLayer = 784
    numHiddenNodes = 30
    numOutputNodes = 10
    numEpochs = 300
    miniBatchSize = 10
    learningRate = 0.5

    #Running model functions
    model1(numInputLayer, numHiddenNodes, numOutputNodes, numEpochs,
           miniBatchSize, learningRate)
    model2A(numInputLayer, numHiddenNodes, numOutputNodes, numEpochs,
           miniBatchSize, learningRate)
    model2B()

#Running main/the program
if __name__ == "__main__":  
    main()
