# Import scipy.special for the sigmoid function expit()
from numpy import random
from numpy.random import f
import scipy.special as scispec
import numpy as np
import matplotlib.pyplot as plt

# Neural network class definition
class NeuralNetwork:

    # Init the network, this gets run whenever we make a new instance of this class
    def __init__ (self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes 
        self.h_nodes = hidden_nodes 
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes)) 
        self.who = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))       

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scispec.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = np.array(inputs_list, ndmin=2).T 
        targets_array = np.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = np.array(inputs_list, ndmin=2).T
        
        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


#----------------------------------
import os
import sys
file = open(os.path.join(sys.path[0], "mnist_train.csv"), "r")

training_array = np.loadtxt(file, delimiter=",")
trainingTargetsList = training_array[0:len(training_array), 0]
trainingInputsArray = training_array[0:len(training_array), 1:len(training_array[0])] / 255 * 0.99 + 0.01

# hiddenNodes = 800
outputNodes = 10
trainingTargets = np.zeros((len(training_array), outputNodes)) + 0.01
for i in range (0, len(trainingTargetsList)):
    digitIndex = trainingTargetsList[i]
    trainingTargets[i][int(digitIndex)] = 0.99


drawDigitImage = False
# Plot this 2D array as an image, use the grey colour map and don’t interpolate
if( drawDigitImage ):
    inputImageIndex = random.randint(0, len(trainingInputsArray)-1)
    image_array = trainingInputsArray[inputImageIndex].reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()


#####------Testing the neural net------------

# Load the MNIST test samples CSV file into a list
test_data_file = open(os.path.join(sys.path[0], "mnist_test.csv"), "r")
test_data_list = test_data_file.readlines() 
test_data_file.close()


epochs = 19
outputNodesCount = 10 
with open(os.path.join(sys.path[0], "mnist_results.csv"), "w") as fh_out:  
    fh_out.write("Hidden Nodes,Learning Rate,#Epochs,Testing Data Accuracy \n" )
    fh_out.flush()
    numIters = 1
    storeTrainingResults = False

    for hiddenNodesCount in [500]:
        for learningRate in [0.03]:
            trainingDataAccuracies = np.zeros(epochs+1)
            testingDataAccuracies = np.zeros(epochs+1)

            for testingIter in range (0, numIters):
                neunet = NeuralNetwork(len(trainingInputsArray[0]), hiddenNodesCount, outputNodesCount, learningRate)

                for epoch in range(1, epochs+1): 
                    print("Hidden Nodes: ", hiddenNodesCount, " lr: ", learningRate, " Epoch: ", epoch, " #Iter: ", testingIter)
                    for input, target in zip( trainingInputsArray, trainingTargets ):
                        neunet.train(input, target)
                    pass

                    if storeTrainingResults:
                        # Scorecard list for how well the network performs, initially empty
                        trainingDataScorecard = []
                        for input, target in zip( trainingInputsArray, trainingTargets ):
                            # Query the network
                            outputs = neunet.query(input)
                            # The index of the highest value output corresponds to the label 
                            label = np.argmax(outputs)
                            # Append either a 1 or a 0 to the scorecard list
                            correct_label = np.argmax(target)
                            if (label == correct_label):
                                trainingDataScorecard.append(1) 
                            else: 
                                trainingDataScorecard.append(0) 
                                pass
                        pass


                    # Scorecard list for how well the network performs, initially empty
                    TestingDataScorecard = []
     
                    # Loop through all of the records in the test data set
                    for record in test_data_list:
                    # Split the record by the commas
                        all_values = record.split(',') 
                        # The correct label is the first value
                        correct_label = int(all_values[0])
                        # print(correct_label, " Correct label")
                        # Scale and shift the inputs
                        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                        # Query the network
                        outputs = neunet.query(inputs)
                        # The index of the highest value output corresponds to the label 
                        label = np.argmax(outputs)
                        # print(label, " Network label")
                        # Append either a 1 or a 0 to the scorecard list
                        if (label == correct_label):
                            TestingDataScorecard.append(1) 
                        else: 
                            TestingDataScorecard.append(0) 
                            if(epoch == 18):
                                print("Wrong: net predicted - ", label, " but answer is ", correct_label)
                                image_array = all_values[1:]
                                trainingArr = np.zeros(784)
                                for i in range (0, 784):
                                    trainingArr[i] = image_array[i]
                                trainingArr = trainingArr.reshape((28, 28))
                                plt.imshow(trainingArr, cmap='Greys', interpolation='None')
                                plt.show()
                            pass
                        pass

                    if storeTrainingResults:
                        # Calculate the performance score on training data, the fraction of correct answers
                        training_data_scorecard_array = np.asarray(trainingDataScorecard)                    
                        trainingDataAccuracy = (training_data_scorecard_array.sum() / training_data_scorecard_array.size)*100
                        trainingDataAccuracies[epoch] += trainingDataAccuracy

                    # Calculate the performance score on testing data, the fraction of correct answers
                    testing_data_scorecard_array = np.asarray(TestingDataScorecard)
                    testingDataAccuracy = (testing_data_scorecard_array.sum() / testing_data_scorecard_array.size)*100
                    testingDataAccuracies[epoch] += testingDataAccuracy

            for epoch in range(1, epochs+1): 
                if storeTrainingResults:
                    trainingAcc = trainingDataAccuracies[epoch]/numIters
                
                testingAcc = testingDataAccuracies[epoch]/numIters

                if storeTrainingResults:
                    print("Training Set: Neural net with: ", hiddenNodesCount, " hidden nodes and ", learningRate, " learning rate at epoch ", epoch )
                    print("Training Set Performance: = ", trainingAcc, '%')

                print("Testing Set: Neural net with: ", hiddenNodesCount, " hidden nodes and ", learningRate, " learning rate at epoch ", epoch, " Accuracy Rating: ", testingAcc, '%' )

                outstr = str(hiddenNodesCount) + "," + str(learningRate)
                outstr += "," + str(epoch) + ","
                if storeTrainingResults:
                    outstr += str(trainingAcc) + ","
                
                outstr += str(testingAcc) + ","
                        
                fh_out.write(outstr + "\n" )
                fh_out.flush()