from __future__ import print_function, division
import numpy as np
import os
import re

import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

f_train_data = open('../data/spatial/data16_path_train01.txt')
f_test_data = open('../data/spatial/data16_path_test01.txt')


# test set
test_data = []
test_label = []

train_label = np.loadtxt('../data/train_label01.txt')
test_label = np.loadtxt('../data/test_label01.txt')


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def calc_accuracy(pre, y):
    right_num = 0
    for index, i in enumerate(y):
        if i == pre[index]:
            right_num += 1
    return float(right_num)/len(y)


def read_all_feature(path):
    vcmpf_data = []
    for index, line in enumerate(path.readlines()):
        file_list = os.listdir(line.strip())
        conv_data = []
        print(line.strip())
        file_list = sorted(file_list, key=lambda x: (int(re.sub('\D', '', x)), x))
        for conv in file_list:
            if os.path.splitext(conv)[1] == '.npy':
                conv_path = line.strip() + "/" + conv
                # print(conv_path)
                data = np.load(conv_path)
                data = np.transpose(np.amax(data, axis=1).reshape((512, -1)))
                # print data.shape
                data = [v for v in data]
                # print(len(data), len(data[0]))
                conv_data.extend(data)
        # print(len(conv_data[0][0]), type(conv_data))
        conv_data = np.array(conv_data)
        print(conv_data.shape)
        indexandchannels = get_the_channel_dic(conv_data)
        print(indexandchannels)
        data = get_vcmpf_feature(conv_data, indexandchannels)
        print(len(data))
        vcmpf_data.append(data)
    return vcmpf_data


def get_the_channel_dic(conv_data):
    # a = conv_data.max(axis=1)
    b = conv_data.argmax(axis=1)
    indexandch = {}
    for i in range(0, conv_data.shape[0]):
        # print("--->{}".format(i))
        # print(b[i])
        if indexandch.get(b[i], 'null') == 'null':
            indexandch[b[i]] = []
            indexandch[b[i]].append(i)
        else:
            indexandch[b[i]].append(i)
    # print(indexandch)
    return indexandch


def get_vcmpf_feature(origin_data,indexandchannels):
    vcmpf = []
    for i in range(0, 512):
        # print(i)
        xj_index = indexandchannels.get(i)
        vi = np.zeros(512, dtype='double')
        if not xj_index is None:
            # print(xj_index)
            xj = np.take(origin_data, xj_index, axis=0)
            xj = np.array(xj)
            max_for_xj = xj.max(axis=0)
            # print(xj.shape)
            # print(len(max_for_xj))
            for j in range(0, 512):
                vi[j] = max_for_xj[j]
        # print(vi)
        vcmpf.extend(vi)
    # print(len(vcmpf))
    return vcmpf


def train_svm(X, y):
    c = 100  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=c).fit(X, y)
    return svc


if __name__ == '__main__':
    train_data = read_all_feature(f_train_data)
    test_data = read_all_feature(f_test_data)
    print("=========data size==========")
    print("num of the features : {}".format(len(train_data[0])))
    print("train data size {}".format(len(train_data)))
    print("train label size {}".format(len(train_label)))
    print("test data size {}".format(len(test_data)))
    print("test label size {}".format(len(test_label)))
    print("start at train {}".format(get_local_time()))
    print("=========training===========")
    mod = train_svm(train_data, train_label)
    print("start at test {}".format(get_local_time()))
    print("=========testing============")
    print(mod.score(test_data, test_label))
    print("end of all opt {}".format(get_local_time()))


