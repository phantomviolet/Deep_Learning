import numpy as np

# ���̾� ������, ��� ������, �н���, ����ġ, ��� ������ ��ųʸ�
Layer_size = 3
minimum_Node_size = 2
maximum_node_size = 5
Learning_rate = 0.1
input_col = 3
output_col = 1
perceptron_weight = {}
perceptron_node_size = {}

# Ȱ��ȭ �Լ� ����
def activation(x):
    return 1 / (1 + np.exp(-x))

#  �Է�, ��� ���̾� ����
def set_input_layer():
    return np.random.uniform(-0.5, 0.5, (input_col, 1))
def set_output_layer():
    return np.random.uniform(1, 1, (output_col, 1))

# ����ġ ����
def set_procession(now_node_size, next_node_size):
    return np.random.uniform(-0.5, 0.5, (now_node_size, next_node_size))

# ��� ������ ����
def set_node_size():
    perceptron_node_size["W1"] = input_col
    for i in range(0, Layer_size):
        perceptron_node_size["W" + str(i + 1)] = np.random.randint(minimum_Node_size, maximum_node_size) 
    return perceptron_node_size

# ��� ����� ���� ����ġ ����
def add_perceptron():
    for i in range(Layer_size):
        if i == 0:
            perceptron_weight["W" + str(i + 1)] = set_procession(input_col, perceptron_node_size["W" + str(i + 1)])
        elif i == Layer_size - 1:
            perceptron_weight["W" + str(i + 1)] = set_procession(perceptron_node_size["W" + str(i + 1)], output_col)
        else:
            perceptron_weight["W" + str(i + 1)] = set_procession(perceptron_node_size["W" + str(i + 1)], perceptron_node_size["W" + str(i + 2)])    
    return perceptron_weight

# ��İ� �Լ�
def matrix_cal_function(matrix_1, matrix_2):
    return matrix_1 @ matrix_2

perceptron_node_size = set_node_size()
perceptron_weight = add_perceptron()

print(perceptron_weight)
print(perceptron_node_size)