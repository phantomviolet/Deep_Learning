import numpy as np
import random

#class
class Layer:
    def __init__(self, Layer_size, Node_size):
        self.Layer_size = Layer_size
        self.Node_size = Node_size
        self.Node = None

    def set_Node(self, Next_Node_Count):
        self.Node = np.random.uniform(-0.5, 0.5, (Next_Node_Count, self.Node_size))

# function  
# sigmoid function
def sigmoid_function(x):
    result = 1 / (1 + np.exp(-x)) 
    return result

# calculate result used input and Node
def matrix_cal_function(input_matrix, Node_matrix):
    result_matrix = Node_matrix @ input_matrix
    for i in range(result_matrix.shape[0]):
        result_matrix[i, 0] = sigmoid_function(result_matrix[i, 0])
    return result_matrix

# calculate by matrix_cal_function
def neural_calculate(Layer_size, input, Neurons):
    result = input
    for i in range(Layer_size):
        if i == Layer_size - 1:
            return result
        else:
            result = matrix_cal_function(result, Neurons[i].Node)

# neurons 2 ~ 16Layer, 2 ~ 256 Node
def set_neurons(Layer_size):
    New_Neurons = [0] * Layer_size
    for i in range(Layer_size):
        Node_size = random.randrange(2, 257)
        New_Neurons[i] = Layer(Layer_size, Node_size)
    for i in range(Layer_size):
        if i == Layer_size - 1:
            Next_Node_Count = 1
        else:
            Next_Node_Count = New_Neurons[i + 1].Node_size
        New_Neurons[i].set_Node(Next_Node_Count)
    return New_Neurons

# set input value 1 ~ 5
def set_input(first_Layer_Node_Count):
    input = np.zeros((first_Layer_Node_Count, 1))
    for i in range(first_Layer_Node_Count):
        input[i, 0] = random.uniform(1, 5)
    return input

# main
Layer_size = random.randrange(2, 17)
Neurons = set_neurons(Layer_size)
first_Layer_Node_Count = Neurons[0].Node_size
input = set_input(first_Layer_Node_Count)
output = neural_calculate(Layer_size, input, Neurons)
print("result: ", output)