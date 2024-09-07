import numpy as np
import matplotlib.pyplot as plt

#データ準備
x = np.arange(0, 6, 0.1)#ゼロから6まで0.1間隔で作成
y1 = np.sin(x)
y2 = np.cos(x)

#グラフ描画
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos") #cos関数は点線で描画
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()