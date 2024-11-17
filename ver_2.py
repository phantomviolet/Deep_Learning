import numpy as np

# Ȱ��ȭ �Լ� ����
def activation(x):
    return 1 / (1 + np.exp(-x))

#  �Է�, ��� ���̾� ����
def set_input_layer():
    return np.random.uniform(-0.5, 0.5, (1, input_col))
def set_output_layer():
    return np.random.uniform(1, 1, (1, output_col))

# ����ġ ����
def set_procession(now_node_size, next_node_size):
    return np.random.uniform(-0.5, 0.5, (now_node_size, next_node_size))

# ��� ������ ����
def set_node_size():
    node_size["W1"] = input_col
    for i in range(1, Layer_size):
        node_size["W" + str(i + 1)] = np.random.randint(minimum_node_size, maximum_node_size) 
    return node_size

# ��� ����� ���� ����ġ ����
def set_weight():
    for i in range(Layer_size):
        if i != Layer_size - 1:
            node_weight["W" + str(i + 1)] = set_procession(node_size["W" + str(i + 1)], node_size["W" + str(i + 2)])
        else:
            node_weight["W" + str(i + 1)] = set_procession(node_size["W" + str(i + 1)], output_col)         
    return node_weight

# ������ �Լ�
def forward():
    for i in range(Layer_size):
        if i == 0:
            forward_result["A" + str(i + 1)] = np.dot(input_layer, node_weight["W" + str(i + 1)])
        else:
            forward_result["A" + str(i + 1)] = np.dot(forward_result["Z" + str(i)], node_weight["W" + str(i + 1)])
        forward_result["Z" + str(i + 1)] = activation(forward_result["A" + str(i + 1)])
    return forward_result

# ���̾� ������, ��� ������, �н���, ����ġ, ��� ������ ��ųʸ�
Layer_size = 3
minimum_node_size = 2
maximum_node_size = 5
Learning_rate = 0.1
input_col = 3
output_col = 1
node_weight = {}
node_size = {}
forward_result = {}

# ����
input_layer = set_input_layer()
print("input: ", input_layer, "\n")
output_layer = set_output_layer()
print("output: ", output_layer, "\n")
node_size = set_node_size()
print("node size: ", node_size, "\n")
node_weight = set_weight()
print("node weight: ", node_weight, "\n")
forward_result = forward()
print("forward: ", forward_result, "\n")


