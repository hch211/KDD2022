import os
import gc
import time
import math
import datetime
from math import log, floor
from sklearn.neighbors import KDTree
import copy


import numpy as np
import dask
# from distributed import Client
# client = Client(n_workers=6)
# dask.init()
# import ray
# ray.init()
os.environ["MODIN_ENGINE"] = "ray"
import modin.pandas as pd
# import pandas as pd

from pathlib import Path
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import joblib
# import pandas
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import argparse

data = pd.read_csv('wtbdata_245days.csv')

def prep_env():

    parser = argparse.ArgumentParser(description='Long Term Wind Power Forecasting')
    parser.add_argument('--pred_file', type=str, default='./predict.py',
                        help='The path to the script for making predictions')
    parser.add_argument('--start_col', type=int, default=3, help='Index of the start column of the meaningful variables')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--is_debug', type=bool, default=False, help='True or False')
    parser.add_argument('--capacity', type=int, default=134, help="The capacity of a wind farm, "
                                                                  "i.e. the number of wind turbines in a wind farm")
    parser.add_argument('--day_len', type=int, default=144, help='Number of observations in one day')
    parser.add_argument('--lag_n', type=int, default=2)
    parser.add_argument('--file', type=str, default='wtbdata_245days.csv')
    args = parser.parse_args()
    settings = {
        'file': args.file,
        "checkpoints": args.checkpoints,
        "start_col": args.start_col,
        "day_len": args.day_len,
        "capacity": args.capacity,
        "pred_file": args.pred_file,
        "is_debug": args.is_debug,
        "lag_n": args.lag_n
    }
    return settings

def search_lagpatv_index(data):
    for i in range(len(data.columns)):
        if data.columns[i] == 'Patv_lag_1':
            return i

def search_timeidx_index(data):
    for i in range(len(data.columns)):
        if data.columns[i] == 'time_idx':
            return i

def lag_feature(data, feat_columns, li):
    # 在对应的位置设置lag列
    temp_data = data.groupby(['TurbID'])[feat_columns]
    for shift_num in li:
        time1 = time.time()
        use_data = temp_data.shift(shift_num)
        time2 = time.time()
        use_data.columns=[i+"_lag_"+str(shift_num) for i in feat_columns]
        data = pd.concat([data,use_data],axis=1)
        time3 = time.time()
        print('groupby use time:{}, concat use time:{}, lag:{}'.format((time2-time1),(time3-time2),shift_num))
    return data

def dataset_split(data):
    if 'Tmstamp' in data.columns.tolist():
        data = data.drop('Tmstamp',axis=1)

    train_data = data[lambda x: x.time_idx < 30816]
    val_data = data[(data.time_idx>=30816) & (data.time_idx < 34993)]
    test_data = data[lambda x: x.time_idx >= 34993]

    train_feat = train_data.drop('Patv',axis=1)
    train_label = train_data['Patv']
    val_feat = val_data.drop('Patv',axis=1)
    val_label = val_data['Patv']
    test_feat = test_data.drop('Patv',axis=1)
    test_label = test_data['Patv']

    return train_feat, train_label, val_feat, val_label, test_feat, test_label


def lgb_(train_feat, train_label, test_feat, test_label):
    gbm = LGBMRegressor(objective='regression', num_leaves=80, learning_rate=0.07, n_estimators=1000, max_depth=7)
    gbm.fit(train_feat, train_label, eval_set=[(test_feat, test_label)], eval_metric='rmse', early_stopping_rounds=20)
    return gbm


def lgb_test_new(gbm, data, test_feat, test_label, show=False):
    y_pred = gbm.predict(test_feat)
    patv_min = min(data.Patv)
    patv_max = max(data.Patv)
    for i in range(len(y_pred)):
        if y_pred[i] < 0:
            y_pred[i] = 0
        if y_pred[i] > patv_max:
            y_pred[i] = patv_max

    y_pred_clear = []
    test_label_clear = []
    for i in range(len(y_pred)):
        #         if not (test_label.values[i] <= 0):
        y_pred_clear.append(y_pred[i])
        test_label_clear.append(test_label.values[i])


    # 模型评估
    #     print('The rmse of prediction is:', mean_squared_error(test_label, y_pred) ** 0.5)
    pred_rmse = mean_squared_error(test_label_clear, y_pred_clear) ** 0.5
    # print('The rmse of prediction is:', pred_rmse)

    # 模型评估
    pred_mae = mean_absolute_error(test_label_clear, y_pred_clear)
    # print('The mae of prediction is:', pred_mae)

    # 特征重要度
    #     print('{}\nFeature importances:{}'.format([column for column in test_feat], list(gbm.feature_importances_)))


    if show == True:
        plt.figure(figsize=(20,5))
        plt.plot(range(len(y_pred_clear)), y_pred_clear, label='pred')
        plt.plot(range(len(test_label_clear)), test_label_clear, label='label')
        plt.legend()
        plt.show()

    return y_pred_clear,pred_rmse,pred_mae

def opreate_nan(df):
    for feature in df.columns[4:14]:
        print(feature)
        for i in range(len(df)):
            if np.isnan(df.loc[i,feature]):
                if i == 0:
                    df.loc[i,feature] = df.loc[i+1,feature]
                else:
                    df.loc[i,feature] = df.loc[i-1,feature]
    return df
    

def opreate_abnormal(df):
    feature_list = ['Ndir','Wdir']
    for i in range(len(df)):
        if df.loc[i,feature_list[0]]>720 or df.loc[i,feature_list[0]]<-720:
            df.loc[i,feature_list[0]] = df.loc[i-1,feature_list[0]]
        if df.loc[i,feature_list[1]]>180 or df.loc[i,feature_list[1]]<-180:
            df.loc[i,feature_list[1]] = df.loc[i-1,feature_list[1]]
    return df
    
def get_test_feat(test_feat,test_label,time_idx):
    indexNames=test_feat[test_feat['time_idx'] !=time_idx].index
    #     print(indexNames)
    test_feat = test_feat.drop(indexNames)
    test_label= test_label.drop(indexNames)
    return test_feat,test_label

def lag_insert(test_feat, num,index_list): #构造test集函数
    feature_start_index = search_lagpatv_index(test_feat)
    for j in range(1,num+1):

        time_idx_list = [m for m in range(34993,34993+j)]

        print(time_idx_list)

        feature_list = test_feat.columns[feature_start_index+j -1:feature_start_index+j]
        print(feature_list)


        for i in range(len(index_list)):
            for feature in feature_list:
                if test_feat.loc[index_list[i],'time_idx'] in time_idx_list:
                    test_feat.loc[index_list[i],feature] = data.loc[index_list[i]-j,'Patv']

                else:
                    test_feat.loc[index_list[i],feature] = np.nan

    return test_feat


def round_pred(gbm1,test_feat,test_label,pred_1,pred_list,rmse_list,mae_list,index_list): #循环预测函数

    for i in range(34994,34994+287):
        feature_start_index = search_lagpatv_index(test_feat)
        feature_list = test_feat.columns[feature_start_index:]
        flag = 0
        for j in range(len(index_list)): #用预测的值填充lag1-n

            if test_feat.loc[index_list[j],'time_idx'] == i:

                for f in range(len(feature_list)):

                    if (index_list[j]+f) in index_list: #边界条件判断是否溢出

                        #                     print(index_list[j]+flag)
                        test_feat.loc[index_list[j]+f,feature_list[f]] = pred_1[flag]

                flag += 1
        #     print(exp_feat)
        #     break;
        real_test_feat,real_test_label = get_test_feat(test_feat,test_label,i)
        # print('predict:{}/288'.format(i-34993+1))
        pred_1,pred_rmse,pred_mae = lgb_test_new(gbm1, data, real_test_feat, real_test_label)
        # print('----------------------------')
        pred_list.extend(pred_1)
        rmse_list.append(pred_rmse)
        mae_list.append(pred_mae)
    return pred_list,rmse_list,mae_list




