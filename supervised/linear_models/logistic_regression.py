#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:55:06 2018

This snippet is about using logistic regression for classification

@author: pierre
"""


#%% Imports

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import linear_model

#%% Load

cancer = datasets.load_breast_cancer()

c_train = cancer.data[:400]
c_train_label = cancer.target[:400]

c_test = cancer.data[401:]
c_test_label = cancer.target[401:]

#%% Plot

fig, axes = plt.subplots(5, 6, figsize=(5, 6))

for i, ax in enumerate(axes.flatten()):
    data1=cancer.data[:,i]*cancer.target
    data2=cancer.data[:,i]*(1-cancer.target)
    ax.hist(
            [data1[data1!=0], data2[data2!=0]],
            color=['red', 'blue'] 
            )
            
#%% Train

rl = linear_model.LogisticRegression()

rl.fit(c_train, c_train_label)


#%% Assess
res = rl.predict(c_test)

#https://en.wikipedia.org/wiki/Precision_and_recall

print("True positives")
tp = float(sum(res*c_test_label))
print(tp)

print("False positives")
fp = float(sum(res*(1-c_test_label)))
print(fp)

print("True negatives")
tn = float(sum((1-res)*(1-c_test_label)))
print(tn)

print("False negatives")
fn = float(sum((1-res)*c_test_label))
print(fn)

print("\n")

print("Precision")
print(tp/(tp+fp))

print("Recall")
print(tp/(tp+fn))