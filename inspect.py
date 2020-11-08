import os
import cv2
import numpy as np

data_dir = "D:\Datasets\PC-13/v1/"

render = "render/"

for folder in os.listdir(data_dir + render):
    path = data_dir + render + folder + "/pc-14-1-" + folder + "-v1-371.png"
    print(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    print(img.shape)
    cv2.imshow("", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()