import argparse
import os
import shutil
import time
import array
import  numpy as np
import random
import uuid

from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from multigrainedscaner import MultiGrainedScaner
from cascadeforest import CascadeForest

f_train_data = open('data/data32_path_train01.txt')
f_test_data = open('data/data32_path_test01.txt')

#feature_name = '/vf_4clips.fc6-all'
feature_name = '/c3d_l2_norm.fc6'

# train set
train_data = []
train_label = []
# test set
test_data = []
test_label = []

print "--> init data set"
# ==============================================
# ##############init data set ####################
# ===============================================

print "--> train init"

for line in f_train_data.readlines():
	file_path = line.strip('\n')+feature_name
	with open(file_path, 'rb') as fid:
		train_item = np.fromfile(fid, np.double)
		train_data.append(train_item)
	# print len(train_data)

print "--> test init"

for line in f_test_data.readlines():
	file_path = line.strip('\n')+feature_name
	with open(file_path, 'rb') as fid:
		test_item = np.fromfile(fid, np.double)
		test_data.append(test_item)
	# print len(test_data)

train_label = np.loadtxt('data/train_label01.txt')
test_label = np.loadtxt('data/test_albel01.txt')

print "--> train set size " + str(len(train_data)) + "\n" + "--> test set size: " + str(len(test_data))
#print str(len(train_label)) + "   " + str(len(test_label))
print "--> init data finish"
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

def calc_accuracy(pre,y):
    return float(sum(pre==y))/len(y)

# C = 1.0  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
# y_pre = svc.predict(X_test)
# print(calc_accuracy(y_pre,y_test))

rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=10).fit(X_train, y_train)
y_pre_rbf = rbf_svc.predict(X_test)
print(calc_accuracy(y_pre_rbf,y_test))

# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
# y_pre_poly= poly_svc.predict(X_test)
# print(calc_accuracy(y_pre_poly,y_test))
#
# lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
# y_pre_lin = lin_svc.predict(X_test)
# print(calc_accuracy(y_pre_lin,y_test))


# cascade_forest_params1 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features=1,n_jobs=-1).get_params()
# cascade_forest_params2 = RandomForestClassifier(n_estimators=1000,min_samples_split=11,max_features='sqrt',n_jobs=-1).get_params()
#
# cascade_params_list = [cascade_forest_params1,cascade_forest_params2]*2
#

# class ProbRandomForestClassifier(RandomForestClassifier):
#     def predict(self, X):
#         return RandomForestClassifier.predict_proba(self, X)




# RF baseline
# RF = RandomForestClassifier(n_estimators = 4000, oob_score = True, n_jobs = -1,random_state =50,max_features = 50,min_samples_split=20)
# RF.fit(X_train, y_train)
# y_pre = RF.predict(X_test)
# print(calc_accuracy(y_pre,y_test))

# Cascade RandomForest Step
# CascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list,k_fold=3)
# CascadeRF.fit(X_train, y_train)
# y_pre_staged = CascadeRF.predict_staged(X_test)
# test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,y_test), 1, y_pre_staged)
# print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))
