from __future__ import print_function, division
import numpy as np
import os
import re
from struct import *
import time
import warnings
from sklearn import svm

f_train_data = open('../data/spatial/data16_path_train01.txt')
f_test_data = open('../data/spatial/data16_path_test01.txt')


# test set
test_data = []
test_label = []

train_label = np.loadtxt('../data/train_label01.txt')
test_label = np.loadtxt('../data/test_label01.txt')

warnings.filterwarnings('ignore')


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def calc_accuracy(pre, y):
    right_num = 0
    for index, i in enumerate(y):
        if i == pre[index]:
            right_num += 1
    return float(right_num)/len(y)


def read_all_feature(path):
    all_data = []
    for index, line in enumerate(path.readlines()):
        file_list = os.listdir(line.strip().replace("c3d-withucf101", "Conv5b-C3D-FT"))
        conv_data = []
        print("conva--->", line.strip().replace("c3d-withucf101", "Conv5b-C3D-FT"))
        file_list = sorted(file_list, key=lambda x: (int(re.sub('\D', '', x)), x))
        for conv in file_list:
            if os.path.splitext(conv)[1] == '.npy':
                conv_path = line.strip().replace("c3d-withucf101", "Conv5b-C3D-FT") + "/" + conv
                # print(conv_path)
                data = np.load(conv_path)
                data = np.transpose(np.amax(data, axis=1).reshape((512, -1)))
                # print data.shape
                data = [v for v in data]
                # print(len(data), len(data[0]))
                conv_data.append(data)
        # print(len(conv_data[0][0]), type(conv_data))
        conv_data = np.amax(conv_data, axis=0)
        conv_data = conv_data.reshape(len(conv_data) * len(conv_data[0]))
        # print(conv_data)
        all_data.append(conv_data)
    # print(len(data))
    return all_data


def read_fc_max_feature(data_path, label_path):
    train_label = np.loadtxt(label_path)
    all_data = []
    f_open = open(data_path)
    for index, line in enumerate(f_open.readlines()):
        file_list = os.listdir(line.strip().replace("c3d-withucf101","cdc_16_feature"))
        one_video_ft = []
        print("fc cdc--->",line.replace("c3d-withucf101","cdc_16_feature").strip())
        for fc in file_list:
            if os.path.splitext(fc)[1] == '.fc7-1-convdeconv':
                fc_path = line.replace("c3d-withucf101","cdc_16_feature").strip() + "/" + fc
                # print(fc_path)
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                s = unpack("iiiii", file_content[:20])
                m = s[0] * s[1] * s[2] * s[3] * s[4]
                data = []
                for i in range(0, m):
                    start = 20 + (i * 4)
                    d = unpack("f", file_content[start:start + 4])
                    data.append(d[0])
                one_video_ft.append(data)
        one_video_ft = np.max(one_video_ft, axis=0)
        all_data.append(one_video_ft)
    return all_data, train_label


def train_svm(X, y):
    c = 0.01  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=c).fit(X, y)
    return svc


def test_for_prb_fusion(modelconv, modelfc, testconv, testfc):
    all = len(test_label)
    count = 0
    for index, x in enumerate(testconv):
        prb_conv = modelconv.predict_proba(x)
        prb_fc = modelfc.predict_proba(testfc[index])
        prb = np.append(prb_conv, prb_fc,axis=0)
        # print(prb)
        prb = np.mean(prb, axis=0)
        print(prb)
        prb_y = np.argmax(prb)
        print(prb_y)
        print(test_label[index])
        if prb_y + 1 == test_label[index]:
            count = count + 1
    print(count)
    print(all)
    print(count/all)


if __name__ == '__main__':
    # train_data = read_all_feature(f_train_data)
    # test_data = read_all_feature(f_test_data)
    # print("---------------train conva------------------")
    # model_conv = train_svm(train_data, train_label)
    train_x, train_y = read_fc_max_feature('../data/spatial/data16_path_train01.txt', '../data/train_label01.txt')
    test_x, test_y = read_fc_max_feature('../data/spatial/data16_path_test01.txt', '../data/test_label01.txt')
    # print("---------------train fc   ------------------")
    # model_fc = train_svm(train_x, train_label)
    # print("---------------tesing     ------------------")
    # test_for_prb_fusion(model_conv, model_fc, test_data, test_x)
    print("=========data size==========")
    print("num of the features : {}".format(len(train_x[0])))
    print("train data size {}".format(len(train_x)))
    print("train label size {}".format(len(train_y)))
    print("test data size {}".format(len(test_x)))
    print("test label size {}".format(len(test_y)))
    print("start at train {}".format(get_local_time()))
    print("=========training===========")
    mod = train_svm(train_x, train_y)
    print("start at test {}".format(get_local_time()))
    print("=========testing============")
    print(mod.score(test_x, test_y))
    # print("end of all opt {}".format(get_local_time()))