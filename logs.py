import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime

def make_log():
    log = []
    
    pickle_out = open("log.pickle","wb")
    pickle.dump(log, pickle_out)
    pickle_out.close()

def print_log():
    pickle_in = open("log.pickle","rb")
    log = pickle.load(pickle_in)
    print(log)

#
# Log structure:
# Shape: n, 
#
# Content Structure: [[[in-sample-accuracy, out-of-sample-accuracy]], [epochs, batch size, seed], [dataset name, training iteration, datetime]]
#

def show_log(entry_num):
    pickle_in = open("log.pickle","rb")
    log = pickle.load(pickle_in)
    l = log[entry_num]
    
    ls = np.array(l[0])
    isample = ls[:, 0]
    osample = ls[:, 1]
    plt.plot(isample)
    plt.plot(osample)
    plt.title("Model evaluation results")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (in percentages)')
    plt.ylim([0, 1])
    plt.legend(["in-sample", "out-of-sample"], loc ="lower right")
    plt.grid()
    pltsteps = []
    for i in range(len(osample)):
        pltsteps.append(float(i))
    for xy in zip(pltsteps, isample):
        plt.annotate(xy[1], xy=xy, textcoords='data')
    for xy in zip(pltsteps, osample):
        plt.annotate(xy[1], xy=xy, textcoords='data')
    plt.show()
    
    print(l)
    
def plot(res):
    xlen = len(res)
    ls = np.array(res)
    isample = ls[:, 0]
    osample = ls[:, 1]
    pltsteps = []
    for i in range(xlen):
        pltsteps.append(int(i))
    print(pltsteps)
    plt.plot(pltsteps, isample)
    plt.plot(pltsteps, osample)
    plt.title("Model evaluation results")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (in percentages)')
    plt.ylim([0, 1])
    plt.legend(["in-sample", "out-of-sample"], loc ="lower right")
    plt.savefig(("D:\Coding\PC-13 Experiments/results/" + detail + ".png"))
    plt.grid()
    for xy in zip(pltsteps, isample):
        plt.annotate(xy[1], xy=xy, textcoords='data')
    for xy in zip(pltsteps, osample):
        plt.annotate(xy[1], xy=xy, textcoords='data')
    plt.figtext(0.01, 0.01, net_srtatt, fontsize = 7)
    plt.figtext(0.01, 0.98, dtime, fontsize = 6)
    plt.savefig(("D:\Coding\PC-13 Experiments/results/" + detail + ".pdf"))
    plt.show()

def process_all():
    pickle_in = open("log.pickle","rb")
    log = pickle.load(pickle_in)
    length = len(log)
    sublength = len(log[0][0])
    print(length, sublength)
    log = np.array(log)
    res = []
    for i in range(sublength):
        isple = []
        for j in range(length):
            isample = log[j][0][i][0]
            isple.append(isample)
        imean = sum(isple) / len(isple)
        osple = []
        for j in range(length):
            osample = log[j][0][i][1]
            osple.append(osample)
        omean = sum(osple) / len(osple)
        res.append([round(imean, 3), round(omean, 3)])
    print(res)
    plot(res)

make_log()

for i in range(20):
    os.system('python train_model.py')
    print("Train iteration: ", i)

net_srtatt = "1x ( Conv2d(5x5) (out: 32) -> MaxPool2d(2x2) with ReLu ) -> 150 -> 13 | Adam, CrossEntropyLoss"
dtime = str(datetime.datetime.now()).replace(":", "-")
detail = str(20) + "e_b" + str(100) + "_seed" + str(3) + "_PC-13-1__1"

print_log()

# for i in range(5):
#     show_log(i)

process_all()