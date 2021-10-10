import numpy as np
import matplotlib.pyplot as plt

# A single perceptron function 
def perceptron(inputs_list, weights_list, bias):  
    # Convert the inputs list into a numpy array  
    inputs = np.array(inputs_list) 
    
    # Convert the weights list into a numpy array  
    weights = np.array(weights_list) 
    
    # Calculate the dot product 
    summed = np.dot(inputs, weights) 
    
    # Add in the bias  
    summed = summed + bias 
    
    # Calculate output 
    # N.B this is a ternary operator, neat huh?  
    output = 1 if summed > 0 else 0 
    
    return output 
 
# Our main code starts here s
 
# Test the perceptron 

# and  
print("AND")  
weights = [1.0, 1.0] 
bias = -1

# nand  
# print("NAND") 
# weights = [-0.5, -0.5] 
# bias = 1
 
# or
# print("OR") 
# weights = [1.0, 1.0] 
# bias = 0

# nor
# print("NOR") 
# weights = [-1.0, -1.0] 
# bias = 1

weights = [1.0, 1.0]
bias = -0.5

# xor - not that simple

print("Weights: ", weights) 
print("Bias: ", bias)  

# Make a new plot (XKCD style)  
fig = plt.xkcd()

allInputs = [ [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0] ]
for input in allInputs:
    print( "Inputs: ", input)
    result = perceptron(input, weights, bias)
    print("Result: ", result)


    # Add points as scatters - scatter(x, y, size, color) 
    # zorder determines the drawing order, set to 3 to make the  
    # grid lines appear behind the scatter points  
    colorToShow = "red"
    if result == 1.0:
        colorToShow = "green"
    
    plt.scatter(input[0], input[1], s=50, color=colorToShow, zorder=3)

    
# Set the axis limits  
plt.xlim(-0.5, 2)
plt.ylim(-0.5, 2)

# Label the plot  
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("State Space of Input Vector")

# Turn on grid lines -c
plt.grid(True, linewidth=1, linestyle=':')

# Autosize (stops the labels getting cut off)  
plt.tight_layout()

x = np.linspace(-0.5, 2)
plt.plot(x, -x * weights[0] / weights[1] - bias / weights[1], 'b', lw=1, label='linear seperator')

# Show the plot  
plt.show()



 
