# Import scipy.special for the sigmoid function expit()
import scipy.special as scispec
import numpy as np

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

#-------------------------Task 4-------------------------------- - Neural net for 2-bit AND/XOR/NAND... logical gates#
# increasing the number of hidden inputs and lowering the learning rate makes the training model more effective and gives it the aability to not be caputred in a local minima
neunet = NeuralNetwork(2, 4, 1, 0.03)

inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
targets = [0.0, 1.0, 1.0, 0.0]

print( "Pre-training:" )
for test_input in inputs:
    print( neunet.query( test_input ) )

error = 4.00
numEpochs = 0
while error > 0.10 * pow(0.1, 2):    
    error = 0.00
    for input, target in zip( inputs, targets ):
        neunet.train( input, target )
        currentResult = neunet.query( input )
        error += 0.5 * pow(currentResult - target, 2)
    numEpochs += 1
    error = error / len(inputs)
    # print(numEpochs, error, neunet.wih, neunet.who)
    if numEpochs > 100000 :
        break



print("\nAfter training")
for test_input, test_target in zip (inputs, targets):
    print( test_input, ": ", neunet.query( test_input ), " result should be ", test_target )

print("\n")


#4 WAY XOR GATE#
print("Task 6")

neunet2 = NeuralNetwork(4, 8, 1, 0.03)

inputs2 = [[0.0, 0.0 , 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [1.0, 0.0 , 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
targets2 = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

print( "Pre-training:" )
for test_input in inputs2:
    print( neunet2.query( test_input ) )

error = 4.00
numEpochs = 0
while error > 0.5 * pow(0.1, 2):    
    error = 0.00
    for input, target in zip( inputs2, targets2 ):
        neunet2.train( input, target )
        currentResult = neunet2.query( input )
        error += 0.5 * pow(currentResult - target, 2)
    numEpochs += 1
    error = error / len(inputs)
    # print(numEpochs, error, neunet.wih, neunet.who)
    if numEpochs > 100000 :
        break


print("\nAfter training")

for test_input, test_target in zip (inputs2, targets2):
    print( test_input, ": ", neunet2.query( test_input ), " = ", test_target )