import math
import random
import numpy as np

# class
class Layer:
    def __init__(self, Layer_Count, Node_Count):
        self.Layer_Count = Layer_Count
        self.Node_Count = Node_Count
        # Node_Count row 1 col
        self.Node = np.zeros((self.Node_Count, 1))
        # self.Node matrix 0,1 ~ 1
        for i in range(self.Node_Count):
            self.Node[i, 0] = random.uniform(0.1, 1.0)
            
# function  
def sig_function(x):
    result = 1 / (1 + math.exp(-x))
    return result

def matrix_cal_function(Node_1, Node_2):
    result_matrix = np.dot(Node_1, Node_2)
    return result_matrix

# neurons 2 ~ 16Layer, 2 ~ 256 Node
def set_neurons(Layer_Count):
    New_Neurons = [0] * Layer_Count
    for i in range(Layer_Count):
        Node_Count = random.randrange(2, 257)
        New_Neurons[i] = Layer(Layer_Count, Node_Count)
    return New_Neurons

# main
Layer_Count = random.randrange(2, 17)
Neurons = [0] * Layer_Count
Neurons = set_neurons(Layer_Count)
first_Layer_Node_Count = Neurons[0].Node_Count
input = np.zeros((first_Layer_Node_Count, 1))
for i in range(first_Layer_Node_Count):
    input[i, 0] = random.randrange(1, 5)

