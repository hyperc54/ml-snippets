#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:15:36 2018

This snippet creates a simple dataset that a Decision Tree model
should model nicely.

The density of points can be adjusted (N)

There's also code for prediction, decision surfaces plotting, and
a visualisation of the model.

@author: pierre
"""
import random as rnd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import tree


#%% Create simple dataset / 4 filled squares, 2 per label
columns = ('x', 'y', 'label')

N = 200

random_half = lambda : rnd.random()/2
data = \
    [(random_half(), random_half(), 1) for i in range(N)] \
    + [(random_half() + 0.5, random_half() + 0.5, 1) for i in range(N)] \
    + [(random_half(), random_half() + 0.5, 2) for i in range(N)] \
    + [(random_half() + 0.5, random_half(), 2) for i in range(N)]

#%% Load dataframe
df = pd.DataFrame(data=data, columns=columns)

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