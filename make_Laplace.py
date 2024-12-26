import os
import numpy as np
import matplotlib.pyplot as plt
import json


h, w = 2400, 2880
SigmaD = 50

heatmap = np.zeros((h, w))
coordinateMtx = np.zeros((h, w, 2))
coordinateMtx[:, :, 0] = np.tile(np.arange(h).reshape(h, 1), (1, w))
coordinateMtx[:, :, 1] = np.tile(np.arange(w), (h, 1))

# Laplacian Filter
def Laplace(center, sigmaD=None):
    center_x, center_y = center
    center_mtx = np.ones((h, w, 2))
    center_mtx[:, :, 0] *= center_y
    center_mtx[:, :, 1] *= center_x

    distance = np.sum(np.abs(coordinateMtx - center_mtx), axis=2)

    LPMap = -(1 / (np.pi * sigmaD**4)) * (1 - (distance**2) / (2 * sigmaD**2)) * np.exp(-distance**2 / (2 * sigmaD**2))
    LPMap = LPMap / np.max(np.abs(LPMap))

    return LPMap

if __name__ == "__main__":
    # json file load
    with open ("train-gt.json", "r", encoding="UTF-8") as train:
        data = json.load(train)

        # json 중 "points" 부분만 추출
        pixel = data["points"]
        for point in pixel:
            label = point['point']
            center_x = label[0]
            center_y = label[1]

            class_num = point["name"]
            img_num = label[2]

            # Make Heatmap
            center = (center_x, center_y)
            heatmap = Laplace(center, sigmaD=SigmaD)

            # Heatmap Plot
            plt.figure(figsize=(10, 10))
            plt.imshow(heatmap, cmap='gist_gray', interpolation='nearest')
            plt.colorbar(label='Label')
            plt.title('Laplace Heatmap')
            plt.xlabel('Coordinate_X')
            plt.ylabel('Coordinate_Y')
            plt.show()