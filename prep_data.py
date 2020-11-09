import os
from tqdm import tqdm
import cv2
import pickle
import numpy as np

base_dir = "D:\Datasets\PC-13/v1/"
sufix = "render/"

train_data = []
shuffeled = []
resz = 150
cat_order = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "ace", "jack", "king", "queen"]

if False:
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

pickle_in = open("train.pickle","rb")
train_data = pickle.load(pickle_in)

#train_data = np.array(train_data)

def shuffle(repetitions=1):
    out = []
    for i in range(0, len(train_data[0]), repetitions):
        for j in range(len(train_data)):
            out.append(train_data[j][i])
    return out

shuffeled = shuffle()
temp_shuffeld = np.array(shuffeled)
print(temp_shuffeld.shape)

file = open("shuffle_patterm.txt","w")
file.truncate(0)
for i in shuffeled:
    file.write(str(i[1]))
file.close()

pickle_out = open("shuffeld.pickle","wb")
pickle.dump(shuffeled, pickle_out)
pickle_out.close()