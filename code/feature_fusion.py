import argparse
import os
import shutil
import array
import  numpy as np
import random
import uuid
import logging
from struct import *

from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from multigrainedscaner import MultiGrainedScaner
from cascadeforest import CascadeForest

logging.basicConfig(filename='feature_fusion_log.txt',level=logging.INFO)
logging.info(time())

f_train_data = open('data/data32_path_train01.txt')
f_train32_data = open('data/data32_path_train01.txt')
f_test_data = open('data/data32_path_test01.txt')

#feature_name = '/vf_4clips.fc6-all'
#feature_name = '/c3d_l2_norm.fc6'

# train set
train_data = []
train_label = []
# test set
test_data = []
test_label = []

train_label = np.loadtxt('data/train_label01.txt')
test_label = np.loadtxt('data/test_albel01.txt')

#all clips feature
train_split_data = []
train_split_label = []

def calc_accuracy(pre,y):
    print sum(pre==y),len(y)
    return float(sum(pre==y))/len(y)

print "init the train set......"
for index,line in enumerate(f_train_data.readlines()):
    file_list = os.listdir(line.strip())
    for fc in file_list:
        if os.path.splitext(fc)[1] == '.fc6-1':
            fc_path = line.strip()+"/"+fc
            # print fc_path
            with open(fc_path,mode='rb') as fid:
                fileContent = fid.read()
            s = unpack("iiiii",fileContent[:20])
            m = s[0]*s[1]*s[2]*s[3]*s[4]
            data = []
            for i in range(0,m):
                start = 20 + (i*4)
                d = unpack("f",fileContent[start:start+4])
                data.append(d[0])
            train_split_data.append(data)
            train_split_label.append(train_label[index])

# print "init the train 32 set......"
# for index,line in enumerate(f_train32_data.readlines()):
#     file_list = os.listdir(line.strip())
#     for fc in file_list:
#         if os.path.splitext(fc)[1] == '.fc6-1':
#             fc_path = line.strip()+"/"+fc
#             # print fc_path
#             with open(fc_path,mode='rb') as fid:
#                 fileContent = fid.read()
#             s = unpack("iiiii",fileContent[:20])
#             m = s[0]*s[1]*s[2]*s[3]*s[4]
#             data = []
#             for i in range(0,m):
#                 start = 20 + (i*4)
#                 d = unpack("f",fileContent[start:start+4])
#                 data.append(d[0])
#             train_split_data.append(data)
#             train_split_label.append(train_label[index])

print len(train_split_data[0]) ,len(train_split_label)

print "start train......"
# C = 1.0  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C,verbose=False).fit(train_split_data, train_split_label)
#RF = RandomForestClassifier(n_estimators = 3000, oob_score = True, n_jobs = -1,random_state =50,max_features = 50,min_samples_split=20)
#RF.fit(train_split_data, train_split_label)
print "Fitting the classifier to the training set"
t0 = time()
param_grid = {'C':[0.1,1,10,100,1000,10000],'gamma':[0.001,0.005,0.01,0.1,10,100,1000]}
clf = GridSearchCV(svm.SVC(kernel='rbf',class_weight = 'balanced'),param_grid)
clf = clf.fit(train_split_data,train_split_label)
print "done in %0.3fs" % (time()-t0)
print "Bets estimator found by grid search:"
print clf.best_estimator_
logging.info(clf.best_estimator_)

print "strat test......"
to = time()
test_pre = []
for index,line in enumerate(f_test_data.readlines()):
	file_list = os.listdir(line.strip())
	test_video_data = []
	for fc in file_list:
		if os.path.splitext(fc)[1] == '.fc6-1':
			fc_path = line.strip()+"/"+fc
			# print fc_path
			with open(fc_path,mode='rb') as fid:
				fileContent = fid.read()
			s = unpack("iiiii",fileContent[:20])
			m = s[0]*s[1]*s[2]*s[3]*s[4]
			test_data = []
			for i in range(0,m):
				start = 20 + (i*4)
				d = unpack("f",fileContent[start:start+4])
				test_data.append(d[0])
			test_video_data.append(test_data)
	#print len(test_video_data)
	y_pre = clf.predict(test_video_data).tolist()
	print y_pre
	print max(y_pre,key=y_pre.count),test_label[index]
	logging.info(y_pre)
	logging.info(max(y_pre,key=y_pre.count))
	logging.info(test_label[index])
	test_pre.append(max(y_pre,key=y_pre.count))
logging.info(classification_report(test_label,test_pre))
logging.info(confusion_matrix(test_label,test_pre))
print(calc_accuracy(test_pre,test_label))
