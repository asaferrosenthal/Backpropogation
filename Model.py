"""
Aaron Safer-Rosenthal | 17asr | 20068164

Backpropogation network which uses gradient descent and sigmoid activation
"""

import random
import numpy as np

class Network():

    #Initializes the network object
    def __init__(self, sizes):

        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = self.setBiases()
        self.weights = self.setWeights()
        
    #Output: Random list of initial biases
    def setBiases(self):
        biases = []
        for y in self.sizes[1:]:
            biases.append(np.random.randn(y, 1))
        return biases

    #Output: Random list of initial weights
    def setWeights(self):
        weights = []
        for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            weights.append(np.random.randn(y, x))
        return weights
    
    #Input: An attribute (bias or weight)
    #Output: A random list of inital values for the attribute
    #Meant to be used during training, rather than during the initalization
    #of the network
    def randomInit(self, attributes):

        randList = []
        for attribute in attributes:
            randList.append(np.zeros(attribute.shape))
        return randList

    #Input: input, a, for network
    #Output: Output of network, activated on input a
    def activation(self, a):

        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    #Inputs: Training data, number of desired epochs, size of mini batches,
    #learning rate and test data (optional)
    #Output: Does not output itself, but calls other functions that update
    #weights and biases
    #Uses gradient descent to train the network.
    #Implements mini batches of the training data to speed up training time,
    #as running each epoch on the entire batch takes a lot longer than
    #mini batches
    def gradientDescent(self, trainingData, numEpochs, miniBatchSize,
                        learningRate, testData):

        trainingData = list(trainingData) #converting from numpy array to
                                          #normal list
        testData = list(testData)
        numTests = len(testData)

        for i in range(numEpochs):
            random.shuffle(trainingData) #randomizing the training data
            miniBatches = self.makeMiniBatches(trainingData, miniBatchSize)
                                            #mini batches speed up training

            for miniBatch in miniBatches:
                self.update(miniBatch, learningRate)
            #self.update_mini_batch(training_data, learning_rate)
                #If you want to use full batch instead of mini-batch
                #(takes a lot longer though)
                
            accuracyCheck = 10 #to see accuracy every x epochs
            if testData and i % accuracyCheck == 0: #printing accuracy
                accuracy = self.evaluate(testData)
                print("Epoch " + str(i) + ": " + str(accuracy) + "/" + \
                      str(numTests))
            else: #to see progress
                print("Epoch " + str(i) + " complete")
            if testData and i == numEpochs-1:
                retLists = []
                retLists.append(self.makeConfusionMatrix(testData))
                retLists.append(self.makeConfusionMatrix(testData))
                return retLists

    #Inputs: training data, and mini batch size
    #Output: Array of mini batches
    #Makes mini batches of data to update the network based on
    def makeMiniBatches(self, data, miniBatchSize):
        
        miniBatches = []
        for i in range(0, len(data), miniBatchSize):
            miniBatches.append(data[i:i+miniBatchSize])
        return miniBatches

    #Input: Mini batch of (training) data, and learning rate
    #Output: Updates network's weights and biases
    #Uses backpropogation and gradient descent, as defined in their respective
    #functions
    def update(self, miniBatch, learningRate):

        b = self.randomInit(self.biases)
        w = self.randomInit(self.weights)

        for x, y in miniBatch:
            vb, vw = self.backpropagation(x, y) #('v' = 'delta') 
            b = self.miniBatchUpdate(b, vb)
            w = self.miniBatchUpdate(w, vw)

        self.weights = self.updateAttribute(self.weights, w, learningRate,
                                            miniBatch)
        self.biases = self.updateAttribute(self.biases, b, learningRate,
                                           miniBatch)

    #Input: Attribute and value changing the attribute
    #Output: A list of the attribute with updated values
    def miniBatchUpdate(self, attribute, vAttribute):
        updateList = []
        for (a, v) in zip(attribute, vAttribute):
            updateList.append(a + v)
        return updateList

    #Input: Current attribute (weight or bias) values, updating attribute
    #values, learning rate and a mini batch of training data
    #Output: Updated vaue for an attribute (weight or bias)
    def updateAttribute(self, selfAttribute, attribute, learningRate, miniBatch):
        updatedAttribute = []
        for (i, n) in zip(selfAttribute, attribute):
            updatedAttribute.append(i - (learningRate / len(miniBatch)) * n)
        return updatedAttribute

    #Input: Features and value for one index of a mini batch
    #Output: Weights and biases for the network
    def backpropagation(self, x, y):

        b = self.randomInit(self.biases)
        w = self.randomInit(self.weights)
        activation = x
        activations = [x] #contains all activations, layer by layer
        xl = [] #contains all x vectors, layer by layer

        for bias, weight in zip(self.biases, self.weights):
            x = np.dot(weight, activation) + bias
            xl.append(x)
            activation = self.sigmoid(x)
            activations.append(activation)

        delta = self.difference(activations[-1], y) *\
        self.sigmoidDerivative(xl[-1])
        b[-1] = delta
        w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.numLayers):
            x = xl[-layer]
            sd = self.sigmoidDerivative(x)
            adjustment = np.dot(self.weights[-layer+1].transpose(), delta) * sd
            b[-layer] = adjustment
            w[-layer] = np.dot(adjustment, activations[-layer-1].transpose())

        return (b, w)

    #Input: Test data
    #Output: Total correct predictions made by the network
    def evaluate(self, testData):

        evaluation = []
        totalCorrect = 0
        for (x, y) in testData:
            evaluation.append([np.argmax(self.activation(x)), y])
            #Argmax returnts the output node with the highest activation
            #(i.e. network's prediction)
        for (x, y) in evaluation:
            totalCorrect += int(x == y) #1 if prediction == actual, 0 if !=
        return totalCorrect

    #Input: Test data
    #Output: Confusion matrix (as a 2D-list)
    def makeConfusionMatrix(self, testData):
        
        evaluation = []
        matrix = matrix = [[0 for _ in range(10)] for _ in range(10)]
        for (sample, result) in testData:
            #print(x,y) #flag
            evaluation.append([np.argmax(self.activation(sample)), result])
        #print(evaluation) #flag
        for (prediction, target) in evaluation:
            """output x-axis, target y-axis
            i.e. row [5,1,2,0] means that it correctly identified 5 1s as 1,
            misidentified 1 2 as 1, 2 3s as 1 and no 4s as 1.
            """
            #print(x,y) #flag
            matrix[prediction][target] += 1
        #print("matrix in function: ", matrix) #flag
        return matrix

    #Input: Confusion matrix (as a 2D-list)
    #Output: Network precision - tp / tp+fp
    def getPrecision(self, matrix):

        totalPrecision = 0
        truePositives = 0
        falsePositives = 0

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i] == matrix[j]:
                    truePositives += matrix[i][j]
                else:
                    falsePositives += matrix[i][j]
            rowPrecision = (truePositives / (truePositives + falsePositives))
            #print("Precision for " + str(i) + ": " + str(rowPrecision)) #flag
            totalPrecision += rowPrecision
            
        totalPrecision /= len(matrix)
        return totalPrecision

    #Input: Confusion matrix (as a 2D-list)
    #Output: Network relevance - tp / tp+fp
    def getRecall(self, matrix):

        totalRecall = 0
        columnList = []
        truePositives = 0
        falseNegatives = 0
        
        for i in range(len(matrix)):
            iColList = []
            for j in range(len(matrix[0])):
                iColList.append(matrix[j][i])
            columnList.append(iColList)

        for i in range(len(columnList)):
            for j in range(len(columnList[0])):
                if i == j:
                    truePositives += columnList[i][j]
                else:
                    falseNegatives += columnList[i][j]
            columnRecall = (truePositives / (truePositives + falseNegatives))
            #print("Relevance for " + str(i) + ": " + str(columnRelevance)) #flag
            totalRecall += columnRecall
            
        totalRecall /= len(matrix)
        return totalRecall

    #Input: Network's output activation and the expected (correct) output
    #Output: The difference between them
    def difference(self, output_activations, y):

        return (output_activations-y)

    #Calculates sigmoid function
    def sigmoid(self, x):

        np.seterr(over = 'ignore') #to avoid the 'RuntimeWarning: overflow
                        #encountered in exp' generated from the next line
        return 1.0/(1.0+np.exp(-x))

    #Calculates derivative of the sigmoid function
    def sigmoidDerivative(self, x):

        return self.sigmoid(x)*(1-self.sigmoid(x))
