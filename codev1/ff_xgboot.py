import numpy as np
import os
from struct import *
from sklearn import svm, preprocessing
import time

# f_train_data = open('../data/spatial/data32_path_train01.txt')
# f_test_data = open('../data/spatial/data32_path_test01.txt')
#
# # train set
# train_data = []
# train_label = []
# # test set
# test_data = []
# test_label = []
#
# train_label = np.loadtxt('../data/train_label01.txt')
# test_label = np.loadtxt('../data/test_label01.txt')
#
# # all clips feature
# train_split_data = []
# train_split_label = []
#
#
# def calc_accuracy(pre, y):
#     print sum(pre == y), len(y)
#     return float(sum(pre == y))/len(y)
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def calc_accuracy(pre, y):
    right_num = 0
    for index, i in enumerate(y):
        if i == pre[index]:
            right_num += 1
    return float(right_num)/len(y)


def read_fc_feature(data_path, label_path):
    train_label = np.loadtxt(label_path)
    f_open = open(data_path)
    train_split_data = []
    train_split_label = []
    for index, line in enumerate(f_open.readlines()):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if os.path.splitext(fc)[1] == '.fc6-1':
                fc_path = line.strip() + "/" + fc
                # print fc_path
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                s = unpack("iiiii", file_content[:20])
                m = s[0] * s[1] * s[2] * s[3] * s[4]
                data = []
                for i in range(0, m):
                    start = 20 + (i * 4)
                    d = unpack("f", file_content[start:start + 4])
                    data.append(d[0])
                train_split_data.append(data)
                train_split_label.append(train_label[index])
    return train_split_data,train_split_label


def read_only_one_featue(data_path, label_path):
    train_label = np.loadtxt(label_path)
    f_open = open(data_path)
    for index, line in enumerate(f_open.readlines()[:100]):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if os.path.splitext(fc)[1] == '.fc6-1':
                fc_path = line.strip() + "/" + fc
                print fc_path
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                s = unpack("iiiii", file_content[:20])
                m = s[0] * s[1] * s[2] * s[3] * s[4]
                data = []
                for i in range(0, m):
                    start = 20 + (i * 4)
                    d = unpack("f", file_content[start:start + 4])
                    data.append(d[0])
                print(data)


def read_avg_feature(data_path, label_path):
    train_label = np.loadtxt(label_path)
    train_data = []
    f_open = open(data_path)
    for index, line in enumerate(f_open.readlines()):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if fc == 'c3d_l2_norm.fc6':
                fc_path = line.strip() + "/" + fc
                # print fc_path
                data = []
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                    # print(len(file_content))
                    for i in range(0, 4096):
                        start = i * 8
                        d = unpack('d', file_content[start:start + 8])
                        data.append(d[0])
                # print(len(data))
                train_data.append(data)
                # print data
                if np.any(np.isnan(data)):
                    print data
                    print fc_path
    return train_data, train_label


def read_16and32_avg_feature(data_path, label_path, data32_path, label32_path):
    train_label = np.loadtxt(label_path)
    train_data = []
    f_open = open(data_path)
    for index, line in enumerate(f_open.readlines()):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if fc == 'c3d_l2_norm.fc6':
                fc_path = line.strip() + "/" + fc
                print fc_path
                data = []
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                    # print(len(file_content))
                    for i in range(0, 4096):
                        start = i * 8
                        d = unpack('d', file_content[start:start + 8])
                        data.append(d[0])
                # print(len(data))
                train_data.append(data)

    train32_label = np.loadtxt(label32_path)
    train32_data = []
    f32_open = open(data32_path)
    for index, line in enumerate(f32_open.readlines()):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if fc == 'c3d.fc6_avg':
                fc_path = line.strip() + "/" + fc
                print fc_path
                data = []
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                    # print(len(file_content))
                    for i in range(0, 4096):
                        start = i * 8
                        d = unpack('d', file_content[start:start + 8])
                        data.append(d[0])
                # print(len(data))
                train32_data.append(data)
    all_data = np.concatenate((train_data, train32_data), axis=1)
    all_label = train_label
    return all_data, all_label


def read_fc_max_feature(data_path, label_path):
    train_label = np.loadtxt(label_path)[:2000]
    all_data = []
    f_open = open(data_path)
    for index, line in enumerate(f_open.readlines()[:2000]):
        file_list = os.listdir(line.strip())
        one_video_ft = []
        print line.strip()
        for fc in file_list:
            if os.path.splitext(fc)[1] == '.fc6-1':
                fc_path = line.strip() + "/" + fc
                # print fc_path
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



def train_model(train_x, train_y):
    c = 0.01  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=c).fit(train_x, train_y)
    # joblib.dump(svc, 'svm_fc_16_concat_32ft16.pkl')
    return svc


def train_boost(train_x, train_y):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=300, learning_rate=0.8)
    res = bdt.fit(train_x, train_y)
    return res


def train_random_forest(train_x, train_y):
    clf = RandomForestClassifier(n_estimators=3000, criterion='entropy', oob_score=True, n_jobs=-1, random_state=1,
                                 max_features=50, min_samples_split=10, min_samples_leaf=1)
    s = clf.fit(train_x, train_y)
    print s
    return s


def train_vote(train_x, train_y):
    # Training classifiers
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[1, 1, 1])
    clf1 = clf1.fit(train_x, train_y)
    clf2 = clf2.fit(train_x, train_y)
    clf3 = clf3.fit(train_x, train_y)
    eclf = eclf.fit(train_x, train_y)
    return clf1,clf2,clf3,eclf


def vote_for_model(model,data_path,label_path):
    test_label = np.loadtxt(label_path)
    f_open = open(data_path)
    all_pre = []
    for index, line in enumerate(f_open.readlines()):
        file_list = os.listdir(line.strip())
        one_video_feature = []
        print line.strip()
        for fc in file_list:
            if os.path.splitext(fc)[1] == '.fc6-1':
                fc_path = line.strip() + "/" + fc
                # print fc_path
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                s = unpack("iiiii", file_content[:20])
                m = s[0] * s[1] * s[2] * s[3] * s[4]
                data = []
                for i in range(0, m):
                    start = 20 + (i * 4)
                    d = unpack("f", file_content[start:start + 4])
                    data.append(d[0])
                one_video_feature.append(data)
        y_pre = model.predict(one_video_feature).tolist()
        print y_pre
        print max(y_pre, key=y_pre.count), test_label[index]
        all_pre.append(y_pre)
    return all_pre,test_label


def train_navie_byais(train_x, train_y):
    gnb = GaussianNB().fit(train_x, train_y)
    return gnb


def get_test_pro(model, test_x, test_y):
    length = len(test_y)
    count = 0
    for index, x in enumerate(test_x):
        prb = model.predict(x)
        print prb
        print test_y[index]
        if prb == test_y[index]:
            count = count + 1
    print count
    print length
    print count/length


if __name__ == '__main__':
    # print("start at read {}".format(get_local_time()))
    # train32_x, train_y = read_fc_max_feature('../data/spatial/data32on16_path_train01.txt', '../data/train_label01.txt')
    # test32_x, test_y = read_fc_max_feature('../data/spatial/data32on16_path_test01.txt', '../data/test_label01.txt')
    train32_x, train_y = read_fc_max_feature('../data/spatial/data32on16_path_train01.txt', '../data/train_label01.txt')
    test32_x, test_y = read_fc_max_feature('../data/spatial/data32on16_path_test01.txt', '../data/test_label01.txt')
    train_x, train_y = read_fc_max_feature('../data/spatial/data16_path_train01.txt', '../data/train_label01.txt')
    test_x, test_y = read_fc_max_feature('../data/spatial/data16_path_train01.txt', '../data/train_label01.txt')

    print(train32_x)
    print(train_x)

    train_x = np.column_stack(train_x, train32_x)
    test_x = np.column_stack(test_x, test32_x)

    train_x = preprocessing.normalize(train_x, norm='l2')
    test_x = preprocessing.normalize(test_x, norm='l2')

    # train_x = np.concatenate((train_x, train32_x), axis=1)
    # test_x = np.concatenate((test_x, test32_x), axis=1)
    # train_x, train_y = read_16and32_avg_feature('../data/spatial/data16_path_train01.txt', '../data/train_label01.txt', '../data/spatial/data32on16_path_train01.txt', '../data/train_label01.txt')
    # test_x, test_y = read_16and32_avg_feature('../data/spatial/data16_path_test01.txt', '../data/test_label01.txt', '../data/spatial/data32on16_path_test01.txt', '../data/test_label01.txt')
    print("=========data size==========")
    print("num of the features : {}".format(len(train_x[0])))
    print("train data size {}".format(len(train_x)))
    print("train label size {}".format(len(train_y)))
    print("test data size {}".format(len(test_x)))
    print("test label size {}".format(len(test_y)))
    print("start at train {}".format(get_local_time()))
    print("=========training===========")
    mod = train_model(train_x, train_y)
    print("start at test {}".format(get_local_time()))
    print("=========testing============")
    # all_pre, test_label = vote_for_model(model,'../data/spatial/data16_path_test01.txt', '../data/test_label01.txt')
    # print mod.score(test_x, test_y)
    get_test_pro(mod, test_x, test_y)
    # get_test_pro(mod,test_x,test_y)
    print("end of all opt {}".format(get_local_time()))
    # #
    # model = joblib.load('svm_fc_16_avg.pkl')
    # pre_y, test_y = vote_for_model(model, '../data/spatial/data16_path_test01.txt', '../data/test_label01.txt')
    # print calc_accuracy(pre_y, test_y)