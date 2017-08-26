import csv
import numpy as np
from sklearn import svm
import time


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_data():
    with open('data_file.csv','r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data


def split_train_test(data):
    train = []
    test = []
    for item in data:
        if item[0] == 'train':
            train.append(item)
        else:
            test.append(item)
    return train, test


def get_class_label():
    res = {}
    with open('classInd.txt') as fin:
        for line in fin.readlines():
            label, classname = line.strip().split(" ")
            if res.get(classname,'null') == 'null':
                res[classname] = label
    return res


def generate_dataset(set, class_and_label):
    data_set = []
    data_label = []
    for index,item in enumerate(set):
        file_path = '/media/zsl/fengzy/sequences/' + item[2] + '-40-features.txt'
        print(file_path)
        data = np.loadtxt(file_path)
        data_label.append(class_and_label[item[1]])
        data = np.amax(data,axis=0)
        print(data.shape)
        data_set.append(data)
    print(data_label)
    return data_set, data_label


def train_model(train_x, train_y):
    c = 0.01  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=c).fit(train_x, train_y)
    # joblib.dump(svc, 'svm_fc_16_concat_32ft16.pkl')
    return svc


if __name__ == '__main__':
    train, test = split_train_test(get_data())
    class_and_label = get_class_label()
    train_x, train_y = generate_dataset(train, class_and_label)
    test_x, test_y = generate_dataset(test, class_and_label)
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
    print mod.score(test_x, test_y)
    print("end of all opt {}".format(get_local_time()))