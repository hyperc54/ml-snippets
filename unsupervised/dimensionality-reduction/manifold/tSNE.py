#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 21:38:28 2018

This snippet is about using t-SNE and playing with it on 3 different datasets.

Great intro from general reduction of dimensionality to t-SNE here:
https://www.youtube.com/watch?v=RJVL80Gg3lA

Snippet inspired from:
https://openclassrooms.com/courses/explorez-vos-donnees-avec-des-algorithmes-non-supervises/decouvrez-une-variete-qui-favorise-la-structure-locale

Good resource to play with t-SNE:
https://distill.pub/2016/misread-tsne/

More info:
https://lvdmaaten.github.io/tsne/

@author: pierre
"""

#%% Imports

# The basics
import numpy as np
import pandas as pd
import random as rnd

# The vis
import matplotlib.pyplot as plt
import seaborn as sns

# The data (3 datasets)
from sklearn.datasets import load_digits, load_iris, fetch_olivetti_faces

# The algorithm
from sklearn import manifold


#%% Load

digits = load_digits()
iris = load_iris()
olivetti = fetch_olivetti_faces()


#%% Global parameters

dataset = olivetti
is_data_image = True # Turn on/off depending on dataset


#%% Visualise data/samples

if is_data_image:
    n_images = 6
    # Visualise random images
    sample = rnd.sample(dataset.images, n_images)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images, 1))
    for i, ax in enumerate(axes):
        ax.imshow(sample[i])
else:
    df = pd.DataFrame(data=dataset.data)
    label = pd.DataFrame(data=dataset.target)
    df['label'] = label
    sns.pairplot(df, hue='label')
    

#%% Perform manifold learning with t-SNE

"""
 - n_components: nb of outputs dimensions
 - n_iter, t-SNE runs a gradient descent over n_iter iterations
 - perplexity, t-SNE tries to keep a local structure, perplexity is roughly
               the number of neighbours we want to take into account for
               determining the position of a given data point.
"""

tsne = manifold.TSNE(n_components=2, perplexity=50, n_iter=3000)
output = tsne.fit_transform(dataset.data)


#%% Visualise output

if is_data_image:
    plot_embedding_images(output)
else:
    plt.scatter(output[:,0], output[:,1], c=dataset.target)
    





#%% Util function
from matplotlib import offsetbox

# Function copied from link above
def plot_embedding_images(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(15, 15))
    ax = plt.subplot(111)

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])
        for i in range(dataset.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 2e-3:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            props=dict(boxstyle='round', edgecolor='white')
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(dataset.images[i], cmap=plt.cm.gray, zoom=0.5), X[i], bboxprops=props)
            ax.add_artist(imagebox)