#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 18:19:30 2018

This snippet is about using the SVM technique without any kernel modifications (yet)
for MultiClass Classification

Good reminders about the optimisation problem that points to the optimal
point and KKT conditions here:
    https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/svm.pdf

@author: pierre

"""

#%% Imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs # to create data
from sklearn import svm

import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

#%%Params
NB_CLUSTERS = 2

#%% load data
X, y = make_blobs(n_samples=40, centers=NB_CLUSTERS, random_state=1)

#%% Visualise
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)


#%% Model
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)


#%%Plot
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

nb_boundaries = nCr(NB_CLUSTERS,2) 
for i in range(nb_boundaries):
    Z = clf.decision_function(xy).reshape(-1,nb_boundaries)[:,i].reshape(XX.shape)
    
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none')
    plt.show()
