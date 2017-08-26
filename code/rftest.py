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

from sklearn import svm
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from multigrainedscaner import MultiGrainedScaner
from cascadeforest import CascadeForest
from sklearn.metrics import confusion_matrix

logging.basicConfig(filename='feature_svm_log_c01_with32.txt',level=logging.INFO)

f_train_data = open('data/data32_path_train01.txt')
f_test_data = open('data/data32_path_test01.txt')
f_train32_data = open('data/data32_path_train01.txt')
f_class_data = open('data/classInd.txt')

#feature_name = '/vf_4clips.fc6-all'
#feature_name = '/c3d_l2_norm.fc6'

# train set
train_data = []
train_label = []
# test set
test_data = []
test_label = []

#class name
class_name = []

train_label = np.loadtxt('data/train_label01.txt')
test_label = np.loadtxt('data/test_albel01.txt')

#all clips feature
train_split_data = []
train_split_label = []

def calc_accuracy(pre,y):
    return float(sum(pre==y))/len(y)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print "init the class name......"
for line in f_class_data.readlines():
	index,name = line.strip().split(" ")
	class_name.append(name)
print "class name len is ",len(class_name)

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
print "init the train 32 set......"
for index,line in enumerate(f_train32_data.readlines()):
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
#            train_split_data.append(data)
#            train_split_label.append(train_label[index])
print len(train_split_data) ,len(train_split_label)

print "start train......"
C = 1  # SVM regularization parameter
svc = svm.SVC(kernel='rbf', C=C,gamma=10).fit(train_split_data, train_split_label)

print "strat test......"
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
	y_pre = svc.predict(test_video_data).tolist()
	print y_pre
	print max(y_pre,key=y_pre.count),test_label[index]
	logging.info(y_pre)
	logging.info(max(y_pre,key=y_pre.count))
	logging.info(test_label[index])
	test_pre.append(max(y_pre,key=y_pre.count))
#cnf_matrix = confusion_matrix(test_label,test_pre)
#plt.figure()
#plot_confusion_matrix(cnf_matrix,classes=class_name,title='Confusion matrix')
#plt.show()
print(calc_accuracy(test_pre,test_label))
