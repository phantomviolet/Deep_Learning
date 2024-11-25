import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 신경망 클래스
class Network:
    def __init__(self, input_image, hidden_size, output_size, minimum_node_size, maximum_node_size):
        self.input_image = input_image
        self.input_image_size = input_image.shape[1]
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.minimum_node_size = minimum_node_size
        self.maximum_node_size = maximum_node_size
        self.input_layer = self.set_input_layer()
        self.node_size = self.set_node_size()
        self.node_weight = self.set_weight()
        self.output_layer = self.set_output_layer()
    
    # 입력 레이어
    def set_input_layer(self):
        return self.input_image
    
    # 노드 사이즈 설정
    def set_node_size(self):
        self.node_size = {}
        self.node_size["W1"] = self.input_image_size
        for i in range(1, self.hidden_size):
            self.node_size["W" + str(i + 1)] = np.random.randint(self.minimum_node_size, self.maximum_node_size) 
        return self.node_size
    
    # 가중치 설정
    def set_weight(self):
        self.node_weight = {}
        for i in range(self.hidden_size):
            if i != self.hidden_size - 1:
                self.node_weight["W" + str(i + 1)] = np.random.uniform(-0.1, 0.1, (self.node_size["W" + str(i + 1)], self.node_size["W" + str(i + 2)]))
            else:
                self.node_weight["W" + str(i + 1)] = np.random.uniform(-0.1, 0.1, (self.node_size["W" + str(i + 1)], self.output_size))         
        return self.node_weight
        
    # 출력 레이어
    def set_output_layer(self):
        return np.zeros((1, self.output_size))
    
    
# 학습 클래스
class Learning:
    def __init__(self, network, target_value, learning_rate):
        self.network = network
        self.target_value = target_value
        self.learning_rate = learning_rate
        self.forward_result = {}
        self.backward_result = {}
        self.deltas = {}
        
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def dleaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
        
    # 순전파
    def forward(self):
        for i in range(self.network.hidden_size):
            if i == 0:
                self.forward_result["A" + str(i + 1)] = np.dot(self.network.input_layer, self.network.node_weight["W" + str(i + 1)])
            else:
                self.forward_result["A" + str(i + 1)] = np.dot(self.forward_result["Z" + str(i)], self.network.node_weight["W" + str(i + 1)])
            self.forward_result["Z" + str(i + 1)] = self.leaky_relu(self.forward_result["A" + str(i + 1)])
        return self.forward_result
    
    # 순전파 결과
    def predict(self):
        return self.forward_result["Z" + str(self.network.hidden_size)]
    
    # 손실함수
    def mse(self):
        return np.mean(np.square(self.target_value - self.predict()))
    
    def dmse(self):
        return self.target_value - self.predict()
    
    # 역전파
    def backward(self):
        # 기울기 계산
        for i in reversed(range(self.network.hidden_size)):
            if i == self.network.hidden_size - 1:
                self.deltas["Delta" + str(i + 1)] = self.dmse() * self.dleaky_relu(self.forward_result["A" + str(i + 1)])
            else:
                self.deltas["Delta" + str(i + 1)] = np.dot(self.deltas["Delta" + str(i + 2)], self.network.node_weight["W" + str(i + 2)].T) * self.dleaky_relu(self.forward_result["A" + str(i + 1)])
        # 가중치 업데이트
        for i in range(self.network.hidden_size):
            if i == 0:
                self.network.node_weight["W" + str(i + 1)] += self.learning_rate * np.dot(self.network.input_layer.T, self.deltas["Delta" + str(i + 1)])
            else:
                self.network.node_weight["W" + str(i + 1)] += self.learning_rate * np.dot(self.forward_result["Z" + str(i)].T, self.deltas["Delta" + str(i + 1)])
        return self.network.node_weight
    
# png 파일 불러오기
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = np.array(image)
    FlattenImage = image.flatten()
    TranspositionImage = FlattenImage.reshape(1, FlattenImage.shape[0])
    return TranspositionImage

image_array = {}

for i in range(1, 6):
    image_path = f"C:/Deep_Learning.worktrees/main/image_{i}.png"
    image_array[f"image_{i}"] = load_image(image_path)
print("main start\n====================================\n")

    
def print_epoch():
    # print("input: ")
    # print(learning.network.input_layer)
    # print("\n====================================\n")
    # print("target: ")
    # print(learning.target_value)
    # print("\n====================================\n")
    print("forward: ")
    print(learning.forward())
    print("\n====================================\n")
    print("predict: ")
    print(learning.predict())
    print("\n====================================\n")
    print("Predict size: ")
    print(learning.predict().shape)
    print("\n====================================\n")
    print("Mean Squared Error: ")
    print(learning.mse())
    print("\n====================================\n")
    print("Initial weight: ")
    print(learning.network.node_weight)
    print("\n====================================\n")
    print("Updated weight: ")
    print(learning.backward())
    
def epoch():
    learning.forward()
    learning.predict()
    learning.mse()
    learning.backward()


    
network = Network(  input_image=image_array["image_5"], 
                    hidden_size=3, 
                    output_size=image_array["image_1"].shape[1], 
                    minimum_node_size=3, 
                    maximum_node_size=5,
)

learning = Learning(network,
                    target_value=image_array["image_1"],
                    learning_rate=0.001
)

count = 0
print_epoch()
print("\n====================================\n")
print_epoch()


# print_epoch()
# 학습

# main
# count = 0
# while True:
#     epoch()
#     count += 1
#     if learning.mse() < 0.001:
#         print("predict: ", learning.predict())
#         print("\n====================================\n")
#         print("Mean Squared Error: ", learning.mse())
#         # print("\n====================================\n")
#         # print("Initial weights: ", network.node_weight)
#         # print("\n====================================\n")
#         # print("Updated Weights: ", learning.backward())
#         print("\n====================================\n")
#         print("Number of iterations: ", count)
#         break
