import os
import random
import matplotlib.pyplot as plt
import datetime
import time
import pickle
from tqdm import tqdm
import numpy as np

from evaluate import eval_accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For reproducibility
seed = 3
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False

IMG_SIZE = 150
class_num = 13
color_channels = 3
dataset_name = "PC-13-1"

X = []
y = []

pickle_in = open("shuffeld.pickle","rb")
shuffeled = pickle.load(pickle_in)

for features, lables in shuffeled:
    X.append(features)
    y.append(lables)

length = len(y)
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)

X = torch.from_numpy(X)
X = X.view(-1, 3, 150, 150)
X = X.to(torch.float32)
y = torch.from_numpy(y)
y = y.to(torch.int64)
print(X.shape, y.shape)

# --- ---
# GPU Test
# --- ---

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    num_gpu = torch.cuda.device_count()
    print("Running on the GPU, device count: ", num_gpu)
else:
    device = torch.device("cpu")
    print("Running on the CPU")

val_size = int(length*0.1)
X = X[:-val_size]
y = y[:-val_size]
Xval = X[-val_size:]
yval = y[-val_size:]

# print("X: ", X.shape, X.dtype)
# print("y: ", y.shape, y.dtype)
# print("Xval: ", Xval.shape, Xval.dtype)
# print("yval: ", yval.shape, yval.dtype)

# --- ---
# Define Network
# --- ---
net_srtatt = "2x ( Conv2d(5x5) (out: 32, 64) -> MaxPool2d(2x2) with ReLu ) -> 512 -> " + str(class_num) + " | Adam, CrossEntropyLoss"

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(3, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        #self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window

        x = torch.randn(IMG_SIZE,IMG_SIZE,3).view(-1,3,IMG_SIZE,IMG_SIZE)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 150) #flattening.
        self.fc2 = nn.Linear(150, class_num) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            print("to linear: ", self._to_linear)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return x #F.softmax(x, dim=1)

net = Net().to(device)
print(net)

optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

BATCH_SIZE = 100
EPOCHS = 20
batch_train_log = []
train_log = []
training_start = datetime.datetime.now()
batch_validation = False

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        try:
            batch_X = X[i:i+BATCH_SIZE]
            batch_y = y[i:i+BATCH_SIZE]
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_data = []

            # -- Batch validation 1 --
            if batch_validation:
                correct = 0
                total = 0
                with torch.no_grad():
                    for i in range(len(batch_X)):
                        real_class = batch_y[i]
                        net_out = net(batch_X[i])[0]  # returns a list, 
                        predicted_class = torch.argmax(net_out)
                        if predicted_class == real_class:
                            correct += 1
                        total += 1
                before_acc = round(correct/total, 3)
                batch_data.append(before_acc)

            # Actual training
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step() # Does the update

            # -- Batch validation 2 --
            if batch_validation:
                correct = 0
                total = 0
                with torch.no_grad():
                    for i in range(len(batch_X)):
                        real_class = batch_y[i]
                        net_out = net(batch_X[i])[0]  # returns a list, 
                        predicted_class = torch.argmax(net_out)
                        if predicted_class == real_class:
                            correct += 1
                        total += 1
                after_acc = round(correct/total, 3)
                batch_data.append(after_acc)
                batch_data = np.array(batch_data)
            batch_train_log.append(batch_data)
        except Exception as e:
            print(" --- AN ERROR OCCURRED ---")
            print("batch_X: ", batch_X.shape)
            print("batch_y: ", batch_y.shape)
            print("current i: ", i)
            try:
                print("outputs: ", outputs[:10])
            except Exception as e1: pass
            print(" ---  ---  --- ")
            print(e)
            quit()

    print(f"Epoch: {epoch}. Loss: {loss}")
    isample, osample = eval_accuracy(X, y, Xval, yval, color_channels, net, IMG_SIZE)
    train_log.append([isample, osample])
    print("In-sample accuracy: ", isample, "  Out-of-sample accuracy: ", osample)

print(train_log)

dtime = str(datetime.datetime.now()).replace(":", "-")

pickle_in = open("log.pickle","rb")
log_out = pickle.load(pickle_in)

logt = np.array(train_log)
outs = []
for i in range(len(logt[:, 0])):
    outs.append([logt[i, 0], logt[i, 1]])
log_out.append([outs, [20, 100, seed], ["PC-13-1", 0, dtime]])

pickle_out = open("log.pickle","wb")
pickle.dump(log_out, pickle_out)
pickle_out.close()