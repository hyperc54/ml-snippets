#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:17:51 2018

This example is about using Linear Regression :
    - in 1D with visualisation
    - in ND

Resources
Trying to understand the intercept value :
    http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-to-interpret-the-constant-y-intercept

@author: pierre
"""

#%% Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets

#%% Load data
diabetes = datasets.load_diabetes()
boston = datasets.load_boston()

#%% Global parameters
dataset = boston
low_d = True # Turn off/on to work in 1D or ND

#%% Select data in case of low_d
if low_d:
    # Col 2 is nice for diabetes, and 5 for boston
    final_data = dataset.data[:, 5]
else:
    final_data = dataset.data

#%% Visualise if low-D
if low_d:
    plt.scatter(final_data, dataset.target) 
    
#%% Perform linear regression
lr = linear_model.LinearRegression()
lr.fit(final_data.reshape(-1,1), dataset.target)

#%% Generate
predictions = lr.predict(final_data.reshape(-1,1))

#%% Evaluate
print("R2 score")
print(lr.score(final_data.reshape(506,1), dataset.target))
print(" or:")
print(r2_score(dataset.target, predictions))

print("\n")
print("Coefficients :")
print(lr.intercept_)
print(lr.coef_)

print("\n")
print("MSE")
print(mean_squared_error(dataset.target, predictions))
    

#%% Visualise
if low_d:
    plt.scatter(final_data, dataset.target,  color='black')
    plt.plot(final_data, predictions, color='blue', linewidth=3)

