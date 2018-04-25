# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 20:53:48 2018

Inspired by:
    https://www.kaggle.com/joaopmpeinado/talkingdata-xgboost-lb-0-966

Kaggle competition:
    https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

Description:
    This kernel is an example of xgboost use in the context
    of a binary classification problem (from a Kaggle competition)

Some info about installing xgboost:
    # http://xgboost.readthedocs.io/en/latest/build.html
    # $ pip install xgboost

@author: pierre
"""


#%% IMPORTS

# Standard imports
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import calendar
import time
import gc

# Specific imports
from sklearn.cross_validation import train_test_split

# xgboost
import xgboost as xgb



#%% Initialise time to measure script duration
start_time = time.time()



#%% GLOBAL VARIABLES

# /!\ Heavy file
# Bash cmd to sample random lines out of big file
# tail -n +2 <file> | shuf -n <nrows> -o <newfile> && sed -i '1i<header>' <newfile>

path = '../input/'
filename = "train.csv"



#%% LOAD TABLE

# Ingest formats
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train_columns = ['ip',
                 'app',
                 'device',
                 'os',
                 'channel',
                 'click_time',
                 'is_attributed']

test_columns = ['ip',
                'app',
                'device',
                'os',
                'channel',
                'click_time',
                'click_id']

# /!\ Specific here, we sampled lines out of the csv
train = pd.read_csv("train.csv",
                    skiprows=range(1,2),
                    nrows=61000,
                    usecols=train_columns,
                    dtype=dtypes)

test = pd.read_csv("test.csv",
                    usecols=test_columns,
                    dtype=dtypes)


print('[{}] Finished to load tables'.format(time.time() - start_time))

#%% FEATURE ENGINEERING

# 
def compute_features(df, is_training=False):
    # Add occurences number
    df_ip_count = df.groupby('ip')['channel'].count().reset_index()
    df_ip_count.columns = ['ip', 'ip_count']

    df_app_os_count = df.groupby(['app', 'os'])['channel'].count().reset_index()
    df_app_os_count.columns = ['app', 'os', 'app_os_count']

    df = pd.merge(df, df_ip_count, on='ip', how='inner', sort=False)
    df = pd.merge(df, df_app_os_count, on=['app','os'], how='inner', sort=False)
    
    # Add time identifiers (here 1mn buckets)
    df['dt'] = pd.to_datetime(df['click_time'])
    df['ts'] = df['dt'].values.astype(np.int64) // 10 ** 9
    df['min_id'] = df['ts'] // 60
    
    df_min_id_count = df.groupby('min_id')['channel'].count().reset_index()
    df_min_id_count.columns = ['min_id', 'min_id_count']

    df = pd.merge(df, df_min_id_count, on=['min_id'], how='inner', sort=False)

    # Compute feature
    df['app_os_by5_pct'] = df['app_os_count'] / df['min_id_count']

    if is_training:
        # Select features
        new_df = df[['app', 'channel', 'ip_count', 'app_os_count', 'app_os_by5_pct', 'is_attributed']]
    else:
        new_df = df[['app', 'channel', 'ip_count', 'app_os_count', 'app_os_by5_pct']]
    
    return new_df

new_train = compute_features(train, is_training=True)

new_test = compute_features(test)
new_test['click_id'] = test['click_id']

print('[{}] Finished to compute features'.format(time.time() - start_time))



#%% TRAIN MODEL
y = new_train['is_attributed']
new_train.drop(['is_attributed'], axis=1, inplace=True)

del train
gc.collect()

#%%
# Set the params for xgboost model (copied from link)
params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True}

dtrain = xgb.DMatrix(new_train, y)


watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)

del new_train
gc.collect()

print('[{}] Finish XGBoost Training'.format(time.time() - start_time))

#%%

# Plot the feature importance from xgboost
xgb.plot_importance(model)
plt.gcf().savefig('feature_importance_xgb.png')

#%%

# Create result dataframe
sub = pd.DataFrame()
sub['click_id'] = new_test['click_id'].astype('int')

new_test.drop(['click_id'], axis=1, inplace=True)
dtest = xgb.DMatrix(new_test)

# Predict
sub['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_sub1.csv', float_format='%.8f', index=False)