#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:06:18 2018

This snippet uses XG boost to solve Titanic Kaggle challenge

Results after submission are far from expectations from the validation sets
IT still needs to be fixed
Plus, Name field is dropped however seing other kernel out there it
can be used to extract the Title which is interesting prediction-wise

@author: pierre
"""

## Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import time
start_time = time.time()

## Load
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


## Feature Engineering
# Create categories out of Cabin, drop room number, keep floor
def format_cabin(cabin_id):
    if cabin_id!=cabin_id:
        return 'NoCabin'
    else:
        return 'Cabin'

def compute_feature_df(df):
    # We simplify cabin entry and format NaN value as a category
    df['cabin_formatted'] = df['Cabin'].apply(format_cabin)
    
    # Family size is computed from Amount of Sibling/Spouse and amount of parent/children
    df['family_size'] = df['SibSp'] + df['Parch']
    
    # There shouldn't be any causation between the fact that a person has not his age registered and his likeliness of survival
    # Hence it feels safe to put the median
    df['age_formatted'] = df['Age'].fillna(df['Age'].median())
    
    # One-hot encode categorical variables
    df[['C1', 'C2', 'C3']] = pd.get_dummies(df['Pclass'], columns=['C1','C2','C3'])
    df[['is_M', 'is_F']] =  pd.get_dummies(df['Sex'], columns=['is_M', 'is_F'])
    df[['Cabin', 'NoCabin']] =  pd.get_dummies(df['cabin_formatted'], columns=['Cabin', 'NoCabin'])
    
    return df

train_target = train['Survived']
test_id = pd.DataFrame(test['PassengerId'])

# We don't care about the embark field (should not be related to chance of survival), same for Ticket number and Name
# We also remove redundant columns
train = compute_feature_df(train)[['PassengerId', 'Survived', 'C1', 'C2', 'C3', 'is_M', 'age_formatted', 'family_size', 'Fare', 'Cabin']]
test = compute_feature_df(test)[['PassengerId', 'C1', 'C2', 'C3', 'is_M', 'age_formatted', 'family_size', 'Fare', 'Cabin']]

X = train.drop(['PassengerId','Survived'], axis=1)
y = train['Survived']


## Model
# Set the params (to tune)
print('{} - Start XGB Training'.format(time.time() - start_time))

params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel': 0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'error', 
          'nthread':8,
          'random_state': 99, 
          'silent': True}
          
# split with 10% validation
x1, x2, y1, y2 = train_test_split(X, y, test_size=0.05, random_state=99)
dtrain = xgb.DMatrix(x1, y1)
dvalid = xgb.DMatrix(x2, y2)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
model = xgb.train(params, dtrain, 200, watchlist, maximize=False, early_stopping_rounds = 50, verbose_eval=5)

print('{} - Finish XGB Training'.format(time.time() - start_time))


## Visualise
plot_importance(model)
plt.gcf().savefig('feature_importance_xgb.png')


## Predict
dtest = xgb.DMatrix(test.drop('PassengerId', axis=1))

## Save
res = pd.DataFrame(test['PassengerId'])
res['Survived'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
res['Survived'] = res['Survived'].apply(lambda x:int(round(x,0)))
res.to_csv('xgb_survived.csv', float_format='%.8f', index=False)