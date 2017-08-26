from __future__ import print_function, division
import numpy as np
import os
import re
import warnings
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

f_train_data = open('../data/spatial/data32on16_path_train01.txt')
f_test_data = open('../data/spatial/data32on16_path_test01.txt')


# test set
test_data = []
test_label = []

train_label = np.loadtxt('../data/train_label01.txt')
test_label = np.loadtxt('../data/test_label01.txt')

warnings.filterwarnings('ignore')


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_time_us():
    return lambda: int(round(time.time() * 1000))


def calc_accuracy(pre, y):
    right_num = 0
    for index, i in enumerate(y):
        if i == pre[index]:
            right_num += 1
    return float(right_num)/len(y)


def read_all_feature(path):
    all_data = []
    for index, line in enumerate(path.readlines()):
        file_list = os.listdir(line.strip())
        conv_data = []
        print(line.strip())
        # print('video--->', int(round(time.time() * 1000)))
        # print('video--->', get_local_time())
        file_list = sorted(file_list, key=lambda x: (int(re.sub('\D', '', x)), x))
        for conv in file_list:
            if os.path.splitext(conv)[1] == '.npy':
                conv_path = line.strip() + "/" + conv
                print(conv_path)
                # print('cpnv--->', int(round(time.time() * 1000)))
                # print('cpnv--->', get_local_time())
                data = np.load(conv_path)
                data = np.transpose(np.amax(data, axis=1).reshape((512, -1)))
                # print data.shape
                data = [v for v in data]
                # print(len(data), len(data[0]))
                conv_data.append(data)
        # print(len(conv_data[0][0]), type(conv_data))
        conv_data = np.amax(conv_data, axis=0)
        # print(conv_data.shape)
        # if line.strip() == '/home/zsl/dataset/c3d-withucf101/TableTennisShot/v_TableTennisShot_g16_c07':
        #     print(conv_data)
        # print(len(conv_data), len(conv_data[0]))
        conv_data = conv_data.reshape(len(conv_data) * len(conv_data[0]))
        # print(conv_data)
        all_data.append(conv_data)
    # print(len(data))
    return all_data


def train_svm(X, y):
    c = 0.01  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=c).fit(X, y)
    return svc


def get_test_pro(model, test_x, test_y):
    for index, x in enumerate(test_x):
        prb = model.predict_proba(x)
        prb = prb.argmax(axis=1)
        print(prb)
        pre_y = model.predict(x)[0]
        print(pre_y)
        print(test_y[index])


def train_random_forest(train_x, train_y):
    clf = RandomForestClassifier(n_estimators=3000, oob_score=True, n_jobs=-1, random_state=1,
                                 max_features=50, min_samples_split=10, min_samples_leaf=1)
    s = clf.fit(train_x, train_y)
    print(s)
    return s

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
    # get_test_pro(mod, test_data, test_label)
    print("end of all opt {}".format(get_local_time()))


