import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 신경망 클래스
class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = self.initialize_weights(input_size, hidden_size)
        self.weights_hidden_output = self.initialize_weights(hidden_size, output_size)
    
    def initialize_weights(self, input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
    
    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden)
        self.hidden_output = self.leaky_relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = self.softmax(self.final_input)
        return self.final_output
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def backward(self, x, y, output, learning_rate):
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.dleaky_relu(self.hidden_input)
        
        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
        self.weights_input_hidden -= learning_rate * np.dot(x.T, hidden_error)
    
    def dleaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output, learning_rate)
            if epoch % 100 == 0:
                loss = self.mse(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

# PNG 파일 불러오기
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((64, 64))
    image_array = np.array(image)
    image_flattened = image_array.flatten()
    return image_flattened

# 이미지 불러오기
image_array = {}
for i in range(1, 6):
    image_path = f"./image_{i}.png"
    image_array[f"image_{i}"] = load_image(image_path)

# 입력 데이터 준비
input_size = 64 * 64
hidden_size = 128
output_size = 5  # 5개의 이미지를 분류한다고 가정

# 입력 데이터와 레이블 준비
x = np.array([image_array[f"image_{i}"] for i in range(1, 6)])
y = np.eye(output_size)  # One-hot encoding으로 레이블 생성

# 신경망 초기화 및 학습
network = Network(input_size, hidden_size, output_size)
network.train(x, y, epochs=1000, learning_rate=0.01)

# 결과 예측
class_labels = ["label_1", "label_2", "label_3", "label_4", "label_5"]  # 각 클래스에 해당하는 단어

for i in range(1, 6):
    output = network.forward(image_array[f"image_{i}"].reshape(1, -1))
    predicted_class = np.argmax(output)
    predicted_label = class_labels[predicted_class]
    print(f"Image {i} predicted class: {predicted_class}, Label: {predicted_label}")

    # 결과 시각화
    plt.imshow(image_array[f"image_{i}"].reshape(64, 64), cmap='gray')
    plt.title(f"Image {i} predicted class: {predicted_class}, Label: {predicted_label}")
    plt.show()