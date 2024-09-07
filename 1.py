import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([10, 20])

#broadcast:形が違う行列でも計算ができる機能
print(x * y)