import os
import numpy as np
import matplotlib.pyplot as plt
import json

# name = Class Number
# Point = Pixel Location (x, y, Img-num)
# Scale = Real Distance

h, w = 2400, 2880
std = 10

heatmap = np.zeros((h, w))
coordinateMtx = np.zeros((h, w, 2)) # Row - Column - Data로 구성
coordinateMtx[:, :, 0] = np.tile(np.arange(h).reshape(h, 1), (1, w))
coordinateMtx[:, :, 1] = np.tile(np.arange(w), (h, 1))

# np.arange(n) = 0 ~ (n-1)까지의 배열 나열
# np.tile(n, repeat_shape) = 배열 n을 repeat_shape 모양이 되도록 반복하여 새로 구성된 배열

def Laplace(center, sigmaD=None):  # Gaussian Filter
    center_x, center_y = center
    center_mtx = np.ones((h, w, 2)) # 전제 값을 1로 지정하여 확률 1.0으로 고정
    center_mtx[:, :, 0] *= center_y
    center_mtx[:, :, 1] *= center_x # 0에 y열 값을, 1에 x열 값을 대입

    distance = np.sum(np.abs(coordinateMtx - center_mtx), axis=2) # Heatmep 제작 시 같은 거리의 값을 통일화시켜줌

    SDMap = (1 / (2 * sigmaD)) * np.exp(-np.abs(distance) / (2 * sigmaD) ** 2)
    SDMap = SDMap / np.max(SDMap)

    return SDMap


if __name__ == "__main__":
    with open ("train-gt.json", "r", encoding="UTF-8") as train:
        data = json.load(train)
        pixel = data["points"]
        # print(pixel)
        for point in pixel:
            # print(point['point'])
            label = point['point']
            # print(label)
            for k in label:
                Laplace(k)


plt.imshow(u, cmap='jet', origin='lower', extent=[0, L, 0, L])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()