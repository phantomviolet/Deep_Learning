import numpy as np

# 레이어 사이즈, 노드 사이즈, 학습률, 가중치, 노드 사이즈 딕셔너리
Layer_size = 3
minimum_Node_size = 2
maximum_node_size = 5
Learning_rate = 0.1
input_col = 3
output_col = 1
perceptron_weight = {}
perceptron_node_size = {}

# 활성화 함수 정의
def activation(x):
    return 1 / (1 + np.exp(-x))

#  입력, 출력 레이어 설정
def set_input_layer():
    return np.random.uniform(-0.5, 0.5, (input_col, 1))
def set_output_layer():
    return np.random.uniform(1, 1, (output_col, 1))

# 가중치 설정
def set_procession(now_node_size, next_node_size):
    return np.random.uniform(-0.5, 0.5, (now_node_size, next_node_size))

# 노드 사이즈 설정
def set_node_size():
    perceptron_node_size["W1"] = input_col
    for i in range(0, Layer_size):
        perceptron_node_size["W" + str(i + 1)] = np.random.randint(minimum_Node_size, maximum_node_size) 
    return perceptron_node_size

# 노드 사이즈에 따른 가중치 설정
def add_perceptron():
    for i in range(Layer_size):
        if i == 0:
            perceptron_weight["W" + str(i + 1)] = set_procession(input_col, perceptron_node_size["W" + str(i + 1)])
        elif i == Layer_size - 1:
            perceptron_weight["W" + str(i + 1)] = set_procession(perceptron_node_size["W" + str(i + 1)], output_col)
        else:
            perceptron_weight["W" + str(i + 1)] = set_procession(perceptron_node_size["W" + str(i + 1)], perceptron_node_size["W" + str(i + 2)])    
    return perceptron_weight

# 행렬곱 함수
def matrix_cal_function(matrix_1, matrix_2):
    return matrix_1 @ matrix_2

perceptron_node_size = set_node_size()
perceptron_weight = add_perceptron()

print(perceptron_weight)
print(perceptron_node_size)