#!/usr/bin/env python
# coding: utf-8


import struct
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_val_predict

import matplotlib.pyplot as plt
import os
import random as rd
import re
import pandas as pd
import itertools
import math



def read_data_svm(path):  #reads the data the same way as CNN but without reshaping
    with open(path) as f:
        i = 0
        data = np.zeros((8000,240))
        for line in f:
            digits = re.findall("\d+\.\d+", line)
            for j in range(len(digits)):
                data[i][j] = pd.to_numeric(digits[j])
            i = i + 1 
            
    return(data)

def gen_labels(size):
    labels = [0]*size
    for i in range(len(labels)):
            labels[i] = math.floor(i/200)
            labels[i] = labels[i] - (math.floor(i/2000)*10)
    return labels

def plot_confusion(y_test, y_pred):
    # Compute and print the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    

def main():
#     os.chdir(os.path.dirname(sys.argv[0]))
    y_pred = None
    y_test = None    

    data = read_data_svm("final.txt")
    labels = gen_labels(len(data))
    X_train, X_test,y_train, y_test = train_test_split(data,labels, 
                                   test_size=0.25, 
                                   shuffle=True)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    c = 5
    gamma = 9.0
    kernel = 'poly'
    print("Kernel = {}".format(kernel))
    print("C = {}".format(c))
    print("Gamma = {}".format(gamma))

    modelSVM = svm.SVC(kernel = 'poly',C=c, gamma = gamma).fit(X_train, y_train)
    
    
    # Use cross-validation to make predictions on the test data
    y_pred = cross_val_predict(modelSVM, X_test, y_test, cv=10)


    # Compute and print the accuracy
    acc = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    
    plot_confusion(y_test, y_pred)

    return

if __name__ == "__main__":
    main()

