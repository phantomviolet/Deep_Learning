import math
import random
import numpy as np
# class
class Layer:
    def __init__(self, Layer_Count, Node_Count):
        self.Layer_Count = Layer_Count
        self.Node_Count = Node_Count
        self.Node = np.zeros((self.Node_Count, 1))
        
    def set_Node(self, Next_Node_Count):
        self.Node = np.random.uniform(0.1, 1.0, (Next_Node_Count, self.Node_Count))
            
# function  
# sigmoid function
def sigmoid_function(x):
    result = 1 / (1 + math.exp(-x))
    return result



# neurons 2 ~ 16Layer, 2 ~ 256 Node
def set_neurons(Layer_Count):
    New_Neurons = [0] * Layer_Count
    for i in range(Layer_Count):
        Node_Count = random.randrange(2, 257)
        New_Neurons[i] = Layer(Layer_Count, Node_Count)
    for i in range(Layer_Count):
        if i == Layer_Count - 1:
            Next_Node_Count = 1
        else:
            Next_Node_Count = New_Neurons[i + 1].Node_Count
        New_Neurons[i].set_Node(Next_Node_Count)
    return New_Neurons

# set input value 1 ~ 4
def set_input(first_Layer_Node_Count):
    input = np.zeros((first_Layer_Node_Count, 1))
    for i in range(first_Layer_Node_Count):
        input[i, 0] = random.randrange(1, 3)
    return input

# calculate result used input and Node
def matrix_cal_function(input_matrix, Node_matrix):
    result_matrix = Node_matrix @ input_matrix
    
    for i in range((result_matrix.ndim)):
        result_matrix[i, 0] = sigmoid_function(result_matrix[i, 0])
    return result_matrix

def neural_calculate(Layer_Count, input):
    result = input
    for i in range(Layer_Count):
        if i == Layer_Count - 1:
            print("result : ", result)
        else:
            result = matrix_cal_function(result, Neurons[i].Node)
# main
Layer_Count = random.randrange(2, 17)
Neurons = [0] * Layer_Count
Neurons = set_neurons(Layer_Count)
first_Layer_Node_Count = Neurons[0].Node_Count
input = set_input(first_Layer_Node_Count)
neural_calculate(Layer_Count, input)


