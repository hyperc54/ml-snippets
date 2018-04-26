#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:13:07 2018

PCA example on MNIST dataset(8x8 images)

Inspired from 
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

@author: pierre
"""

#%% Imports
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#%% Load
digits = load_digits() # Dict

#%% Digits lookup
plt.imshow(digits.images[32])

#%% Init model
k = 2
pca = PCA(n_components=k)

#%% Apply
model = pca.fit(digits.data)

components = model.components_
projected = model.transform(digits.data)

#%% represent axis
plt.imshow(components[0].reshape(8,8))
plt.figure()
plt.imshow(components[1].reshape(8,8))

#%% plot 
plt.figure()
plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()

#%%
pca.get_precision()

#%% 3D version with k=3 
"""
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c=digits.target)
"""

#%% To find the optimal number of components 
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#%% Plot the 10 first original digits and the projected digits
def plot_digits(data):
    fig, axes = plt.subplots(2, 10, figsize=(10, 2),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes[0]):
        ax.imshow(digits.images[i],
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
    for i, ax in enumerate(axes[1]):
        ax.imshow((projected[i, 0]*components[0] + projected[i, 1]*components[1]).reshape(8,8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
plot_digits(digits.data)


