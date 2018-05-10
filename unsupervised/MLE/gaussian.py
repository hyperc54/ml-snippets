#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:50:22 2018

This snippet will perform MLE of gaussian models in 2D

@author: pierre
"""


#%% Imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import mixture

#%% Load
iris = datasets.load_iris()

df_iris = \
    pd.DataFrame(
            np.concatenate(
                    (iris.data, iris.target.reshape(150,-1)),
                    axis=1,
                    ),
            columns=['feat1','feat2','feat3','feat4','species']
            )


#%% Visualise
sns.pairplot(df_iris, hue='species')

#%% Fit model on composition of features

clf = mixture.GaussianMixture(n_components=3)
clf.fit(df_iris[['feat1','feat3']])

#%% Predict and compare with true labels
predictions = clf.predict(df_iris[['feat1','feat3']])

fix, ax=plt.subplots(nrows=1, ncols=2)

ax[0].scatter(df_iris['feat1'], df_iris['feat3'], c=iris.target)
ax[1].scatter(df_iris['feat1'], df_iris['feat3'], c=predictions)

#%% Plot the means

means = clf.means_
covariances = clf.covariances_

ax[1].scatter(means[:,0], means[:,1], c='black', s=100)