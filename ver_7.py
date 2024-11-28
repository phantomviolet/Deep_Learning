import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Leaky ReLU, Softmax, Cross-Entropy Loss
def leakyReLU(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def dleakyReLU(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# 오버플로우 해결 => gpt
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # print(exp_x / np.sum(exp_x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 활성화 함수가 leaky ReLU 일 경우 미분이 단순해지는 cross-entropy loss 이용
def cross_entropy_loss(real, predicted):
    return -np.sum(real * np.log(predicted)) / len(real)

# 가중치 초기화
def initWeight(input_size, layer_sizes):
    weights = []
    biases = []
    layer_input = input_size
    for size in layer_sizes:
        # 허 초기화
        weight = np.random.randn(layer_input, size) * np.sqrt(2 / (layer_input + size))
        bias = np.zeros((1, size))
        weights.append(weight)
        biases.append(bias)
        layer_input = size
    return weights, biases

# 순전파
def forward(inputValue, weights, biases, alpha=0.01):
    after_activation = [inputValue]
    before_activation = []
    for W, b in zip(weights, biases):
        Z = after_activation[-1] @ W + b
        A = leakyReLU(Z, alpha)
        before_activation.append(Z)
        after_activation.append(A)
        
    output = softmax(after_activation[-1])
    # print(output[1])
    # print(output.shape)
    return output, before_activation, after_activation

# 역전파
def backward(predictValue, after_activation, before_activation, weights, targetValue, alpha=0.01):
    d_Weight = []
    d_bias = []
    m = targetValue.shape[0]  # 배치 크기
    
    # 출력 레이어에서의 dZ 계산 (Softmax + Cross-Entropy Loss의 특성 활용)
    dZ = predictValue - targetValue  # Cross-Entropy의 미분
    
    for i in reversed(range(len(weights))):
        if i < len(weights) - 1:
            dZ = dA * dleakyReLU(before_activation[i], alpha)
        
        # 가중치, 편향 업데이트
        dW = (after_activation[i].T @ dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA = dZ @ weights[i].T
        
        # 값 저장
        d_Weight.insert(0, dW)
        d_bias.insert(0, db)
    
    return d_Weight, d_bias


# 이미지 불러오기
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    return image_array

# main
# 이미지 불러오기
image_array = []
for i in range(1, 71):
    imagePath = f"/Users/parksungryeong/projects/deepLearning/Deep_Learning/learning/{i}.png"
    image_array.append(load_image(imagePath))
learning_image = [img.flatten()[np.newaxis, :] for img in image_array]

image_size = 4096 # 64 * 64
output_size = 7 # 현재 구분할 수 있는 사진 개수

# 하이퍼 파라미터
layer_sizes = [10, 10, 10, 10, output_size]  # 마지막 output_size는 출력층으로
learning_rate = 0.001
epochs = 1000


# 출력 원 핫 벡터 1~5, 6~10, 11~15
targetValue = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1],
])

# 모델 초기화
weights, biases = initWeight(image_size, layer_sizes)

predict, before_activation, after_activation = forward(learning_image, weights, biases)

for epoch in range(epochs):
    lossValue = 0
    for img, target in zip(learning_image, targetValue):
        predict, before_activation, after_activation = forward(img, weights, biases)
        d_Weight, d_bias = backward(predict, after_activation, before_activation, weights, target[np.newaxis, :])
        for i in range(len(weights)):
            weights[i] -= learning_rate * d_Weight[i]
            biases[i] -= learning_rate * d_bias[i]

        lossValue += cross_entropy_loss(target[np.newaxis, :], predict) 
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {lossValue:.4f}")

# 이미지 구별
# new_image = load_image("/Users/parksungryeong/projects/deepLearning/Deep_Learning/test/test6.png")
# new_image = new_image.flatten()[np.newaxis, :]
# predict, _, _ = forward(new_image, weights, biases)
# print(predict)
# picture_class = np.argmax(predict)
# if picture_class == 0:
#     print("이미정")
# elif picture_class == 1:
#     print("박대훈")
# elif picture_class == 2:
#     print("박성령")
# elif picture_class == 3:
#     print("이종빈")
# elif picture_class == 4:
#     print("유재화")
# elif picture_class == 5:
#     print("유수연")
# elif picture_class == 6:
#     print("박수진")
    
while True:
    new_image = load_image(input("이미지 경로를 입력해주세요: "))
    new_image = new_image.flatten()[np.newaxis, :]
    predict, _, _ = forward(new_image, weights, biases)
    print(predict)
    picture_class = np.argmax(predict)
    if picture_class == 0:
        print("이미정")
    elif picture_class == 1:
        print("박대훈")
    elif picture_class == 2:
        print("박성령")
    elif picture_class == 3:
        print("이종빈")
    elif picture_class == 4:
        print("유재화")
    elif picture_class == 5:
        print("유수연")
    elif picture_class == 6:
        print("박수진")
    if input("계속하시겠습니까? (y/n): ") == "n":
        break