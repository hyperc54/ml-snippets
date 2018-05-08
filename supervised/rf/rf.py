"""
Created on Tue May  8 21:01:57 2018

Python3

This notebook uses a Random Forest model to solve the titanic challenge on
Kaggle

No tuning is done on the forest, it's the next step to improve results.
(to be done on a validation set)

links:https://www.kaggle.com/hyperc/titanic-decisiontree-rf/
https://www.kaggle.com/c/titanic

@author: pierre
"""

## Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
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
train = compute_feature_df(train)[['PassengerId','Survived','C1', 'C2', 'C3', 'is_M', 'age_formatted', 'family_size', 'Fare', 'Cabin']]
test = compute_feature_df(test)[['PassengerId','C1', 'C2', 'C3', 'is_M', 'age_formatted', 'family_size', 'Fare', 'Cabin']]



#model
"""
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train[['C1', 'C2', 'C3', 'is_M', 'age_formatted', 'family_size', 'Fare', 'Cabin']], train['Survived'])
"""
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train[['C1', 'C2', 'C3', 'is_M', 'age_formatted', 'family_size', 'Fare', 'Cabin']], train['Survived'])

test=test.fillna(test['Fare'].mean())
test['Survived'] = clf.predict(test.drop('PassengerId',axis=1))

test[['PassengerId','Survived']].to_csv('dt_survived.csv', float_format='%.8f', index=False)