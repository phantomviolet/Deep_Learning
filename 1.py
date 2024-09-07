import numpy as np
import matplotlib.pyplot as plt

#データ準備
x = np.arange(0, 6, 0.1)#ゼロから6まで0.1間隔で作成
y = np.sin(x)

#グラフ描画
plt.plot(x, y)
plt.show()