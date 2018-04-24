#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:15:36 2018

@author: pierre
"""
import random as rnd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import tree


#%% Create simple dataset
columns = ('x', 'y', 'label')

sparse_data = \
    [(0.1, 0.2, 1), (0.3, 0.4, 1), (0.4, 0.1, 1), (0.3, 0.3, 1), (0.25, 0.4, 1),
     (0.60, 0.2, 2), (0.78, 0.1, 2), (0.55, 0.4, 2), (0.85, 0.2, 2), (0.75, 0.3, 2),
     (0.65, 0.82, 1), (0.75, 0.75, 1), (0.80, 0.65, 1), (0.90, 0.60, 1), (0.86, 0.7, 1),
     (0.1, 0.65, 2), (0.2, 0.75, 2), (0.25, 0.55, 2), (0.45, 0.85, 2), (0.3, 0.70, 2)]

N = 200
random_half = lambda : rnd.random()/2
dense_data = \
    [(random_half(), random_half(), 1) for i in range(N)] \
    + [(random_half() + 0.5, random_half() + 0.5, 1) for i in range(N)] \
    + [(random_half(), random_half() + 0.5, 2) for i in range(N)] \
    + [(random_half() + 0.5, random_half(), 2) for i in range(N)]

which_data = dense_data



#%% Load dataframe
df = pd.DataFrame(data=which_data, columns=columns)

#%% Plot the data
colors = {1:'red', 2:'blue'}
df['color'] = df['label'].apply(lambda x: colors[x])
df.plot.scatter(x='x', y='y', c=df['color'])

#%% Fit a decision tree model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(df[['x','y']], df['label'])

#%% Test a single prediction
to_pred = (0.2,0.4)
clf.predict([to_pred])



#%% Plot decision surfaces

margin = 0.1
plot_step = 0.01
x_min, x_max = df['x'].min() - margin, df['x'].max() + margin
y_min, y_max = df['y'].min() - margin, df['y'].max() + margin

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

# ... with the training points
plt.scatter(df['x'], df['y'], c=df['color'], edgecolor='black', s=15)



#%% Print the serialised model
# required - graphviz : apt-get install graphviz
# required - pydot : pip install pydot

from sklearn.externals.six import StringIO  
import pydot 

dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue())

graph[0].write_pdf("tree.pdf")

# We can point out that in the dense format, the tree is not as simple as
# expected !! It is due to the fact that the perfect first split (at 0.5)
# doesn't separate the different labels at all. (the algorithm doesn't foresee
# farther than one shot...)