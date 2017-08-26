import argparse
import os
import shutil
import time
import array
import  numpy as np
import random
import uuid
import logging
import itertools
import matplotlib.pyplot as plt
from struct import *


f_train_data = open('../data/spatial/data32on16_path_train01.txt')
f_test_data = open('../data/spatial/data32on16_path_test01.txt')

# train set
train_data = []
train_label = []
# test set
test_data = []
test_label = []


#all clips feature
train_split_data = []
train_split_label = []

def calc_accuracy(pre,y):
    return float(sum(pre==y))/len(y)


def get_file(data):
    for index, line in enumerate(data.readlines()):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if os.path.splitext(fc)[1] == '.conv5b':
                fc_path = line.strip() + "/" + fc
                # print fc_path
                with open(fc_path, mode='rb') as fid:
                    fileContent = fid.read()
                s = unpack("iiiii", fileContent[:20])
                m = s[0] * s[1] * s[2] * s[3] * s[4]
                data = []
                for i in range(0, m):
                    start = 20 + (i * 4)
                    d = unpack("f", fileContent[start:start + 4])
                    data.append(d[0])
                data = np.array(data)
                a = data.reshape((512, 2, 7, 7))
                save_path = fc_path
                print save_path
                np.save(save_path, a)


if __name__ == '__main__':
    get_file(f_train_data)
    get_file(f_test_data)
