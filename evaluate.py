import os
import random
import matplotlib.pyplot as plt
import datetime
import time
import pickle
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def eval_accuracy(X, y, Xval, yval, cc, net, IMG_SIZE):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        num_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
    
    correct = 0
    total = 0
    eval_size = int(len(X)*0.1)
    eval_X = X[:eval_size]
    eval_y = y[:eval_size]
    with torch.no_grad():
        for i in range(len(eval_X)):
            real_class = eval_y[i]
            net_out = net(eval_X[i].view(-1, cc, IMG_SIZE, IMG_SIZE).to(device))[0]  # returns a list
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    in_sample_acc = round(correct/total, 3)
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(Xval)):
            real_class = yval[i]
            net_out = net(Xval[i].view(-1, cc, IMG_SIZE, IMG_SIZE).to(device))[0]  # returns a list, 
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    out_sample_acc = round(correct/total, 3)

    return in_sample_acc, out_sample_acc