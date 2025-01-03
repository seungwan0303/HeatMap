import os
import numpy as np
import matplotlib.pyplot as plt
import json

# name = Class Number
# Point = Pixel Location (x, y, Img-num)
# Scale = Real Distance

h, w = 2400, 2880 # Dimensions of the heatmap
SigmaD = 50

heatmap = np.zeros((h, w))
coordinateMtx = np.zeros((h, w, 2)) # Row - Column - Data로 구성
coordinateMtx[:, :, 0] = np.tile(np.arange(h).reshape(h, 1), (1, w))
coordinateMtx[:, :, 1] = np.tile(np.arange(w), (h, 1))

# np.arange(n) = 0 ~ (n-1)까지의 배열 나열
# np.tile(n, repeat_shape) = 배열 n을 repeat_shape 모양이 되도록 반복하여 새로 구성된 배열

# Gaussian Filter
def Gaussain(center, sigmaD=None):
    center_x, center_y = center
    center_mtx = np.ones((h, w, 2)) # 2400x2880 사이즈의 각 픽셀에 전제 값을 1로 지정하여 확률 1.0으로 고정
    center_mtx[:, :, 0] *= center_y # 모든 칸에 y값을 대입
    center_mtx[:, :, 1] *= center_x # 모든 칸에 x갑슬 대입

    distance = np.sum(np.abs(coordinateMtx - center_mtx), axis=2) # Heatmep 제작 시 열의 관점에서 같은 거리의 값을 통일화시켜줌

    SDMap = (1 / (2 * sigmaD)) * np.exp(-np.abs(distance) / (2 * sigmaD) ** 2)
    SDMap = SDMap / np.max(SDMap)

    return SDMap

# Heatmap Part
if __name__ == "__main__":
    # json file load
    with open ("train-gt.json", "r", encoding="UTF-8") as train:
        data = json.load(train)

        # json 중 "points" 부분만 추출
        pixel = data["points"]
        for point in pixel:
            label = point['point'] # "points" 중에서 좌표값만 추출
            center_x = label[0]
            center_y = label[1]

            class_num = point["name"]  # "points" 중에서 class값만 추출
            img_num = label[2]

            # Make Heatmap
            center = (center_x, center_y)
            heatmap = Gaussain(center, sigmaD=SigmaD)

            # Heatmap Plot
            plt.figure(figsize=(10, 10))
            plt.imshow(heatmap, cmap='gist_gray', interpolation='nearest')
            plt.colorbar(label='Label')
            plt.title('Gaussian Heatmap')
            plt.xlabel('Coordinate_X')
            plt.ylabel('Coordinate_Y')
            plt.show()