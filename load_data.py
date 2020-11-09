import os
from tqdm import tqdm
import cv2
import pickle
import numpy as np

import torch

base_dir = "D:\Datasets\PC-13/v1/"
sufix = "render/"

train_data = []
resz = 150
cat_order = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "ace", "jack", "king", "queen"]

for folder in tqdm(os.listdir(base_dir + sufix)):
    path = base_dir + sufix + folder + "/"
    out = []
    for image_name in os.listdir(path):
        img_in = cv2.imread((path + image_name), cv2.IMREAD_COLOR)
        img_resz = cv2.resize(img_in, (resz, resz))
        class_num = cat_order.index(folder)
        out.append([img_resz, class_num])
    train_data.append(out)

pickle_out = open("train.pickle","wb")
pickle.dump(train_data, pickle_out)
pickle_out.close()

train_data = np.array(train_data)
print(train_data.shape, train_data[0][0][0].shape)