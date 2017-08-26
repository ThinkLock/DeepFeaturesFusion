import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.datasets import fetch_mldata
from multigrainedscaner import MultiGrainedScaner
from cascadeforest import CascadeForest
import gzip, struct

# mnist = fetch_mldata('MNIST original')
#
# # Trunk the data
# n_train = 60000
# n_test = 10000
#
# # Define training and testing sets
# train_idx = np.arange(n_train)
# test_idx = np.arange(n_test)+n_train
# random.shuffle(train_idx)
#
# X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
# X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]

# def _read(image,label):
#     minist_dir = 'MNIST_data/'
#     with gzip.open(minist_dir+label) as flbl:
#         magic, num = struct.unpack(">II", flbl.read(8))
#         label = np.fromstring(flbl.read(), dtype=np.int8)
#     with gzip.open(minist_dir+image, 'rb') as fimg:
#         magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
#         image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
#     return image,label
#
# X_train,y_train = _read('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
# X_test,y_test = _read('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
f_train_data = open('data/data32_path_train01.txt')
f_test_data = open('data/data32_path_test01.txt')

# feature_name = '/vf_3clips.fc6-all'
feature_name = '/c3d_l2_norm.fc6'

# train set
train_data = []
train_label = []
# test set
test_data = []
test_label = []

print "init data set"
# ==============================================
# ##############init data set ####################
# ===============================================
for line in f_train_data.readlines():
	file_path = line.strip('\n')+feature_name
	with open(file_path, 'rb') as fid:
		train_item = np.fromfile(fid, np.double)
		train_data.append(train_item)

for line in f_test_data.readlines():
	file_path = line.strip('\n')+feature_name
	with open(file_path, 'rb') as fid:
		test_item = np.fromfile(fid, np.double)
		test_data.append(test_item)

train_label = np.loadtxt('data/train_label01.txt')
test_label = np.loadtxt('data/test_albel01.txt')

print "train set size " + str(len(train_data)) + "\n" + "test set size: " + str(len(test_data))
#print str(len(train_label)) + "   " + str(len(test_label))
print "init data finish"
# ================================================

# random forest test case
# clf = RandomForestClassifier(n_estimators=1000)
# clf = clf.fit(train_data, train_label)
# score = clf.score(test_data,test_label)
# print score

X_train = np.array(train_data)
y_train = np.array(train_label)
X_test = np.array(test_data)
y_test = np.array(test_label)



scan_forest_params1 = RandomForestClassifier(n_estimators=30,min_samples_split=21,max_features=1,n_jobs=-1).get_params()
scan_forest_params2 = RandomForestClassifier(n_estimators=30,min_samples_split=21,max_features='sqrt',n_jobs=-1).get_params()

cascade_forest_params1 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features=1,n_jobs=-1).get_params()
cascade_forest_params2 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features='sqrt',n_jobs=-1).get_params()

scan_params_list = [scan_forest_params1,scan_forest_params2]
cascade_params_list = [cascade_forest_params1,cascade_forest_params2]*2

def calc_accuracy(pre,y):
    return float(sum(pre==y))/len(y)
class ProbRandomForestClassifier(RandomForestClassifier):
    def predict(self, X):
        return RandomForestClassifier.predict_proba(self, X)

train_size = 10000
# gcForest

# Multi-Grained Scan Step
Scaner1 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./4)
Scaner2 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./9)
Scaner3 = MultiGrainedScaner(ProbRandomForestClassifier(), scan_params_list, sliding_ratio = 1./16)

X_train_scan =np.hstack([scaner.scan_fit(X_train[:train_size], y_train[:train_size])
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])
X_test_scan = np.hstack([scaner.scan_predict(X_test)
                             for scaner in [Scaner1,Scaner2,Scaner3][:1]])

# Cascade RandomForest Step
CascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list)
CascadeRF.fit(X_train_scan, y_train[:train_size])
y_pre_staged = CascadeRF.predict_staged(X_test_scan)
test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged)
print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))

# CascadeRF baseline
BaseCascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list,k_fold=3)
BaseCascadeRF.fit(X_train[:train_size], y_train[:train_size])
y_pre_staged = BaseCascadeRF.predict_staged(X_test)
test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged)
print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))

# RF baseline
RF = RandomForestClassifier(n_estimators=1000)
RF.fit(X_train[:train_size], y_train[:train_size])
y_pre = RF.predict(X_test)
print(calc_accuracy(y_pre,y_test))
