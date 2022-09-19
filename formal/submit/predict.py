import os
import gc
import time
import math
import datetime
from math import log, floor
# from distributed import Client
#
# client = Client()

import copy

import numpy as np
os.environ["MODIN_ENGINE"] = "ray"
import modin.pandas as pd
# import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm as tqdm
from scipy.signal import find_peaks

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import joblib
from prepare import *














def single_step_withwspd(wspd_lag,lag_n):
    # 做lag特征
    data = pd.read_csv('wtbdata_245days_normal_all.csv')
    #data = opreate_nan(data)
    #data = opreate_abnormal(data)
    feat_columns = ['Wspd','Wdir','Pab1']
    li = [i for i in range(288,288+wspd_lag)]
    data_lag1 = lag_feature(data, feat_columns, li)

    feat_columns = ['Patv']
    li = [i for i in range(1,lag_n+1)]
    data_lag1 = lag_feature(data_lag1, feat_columns, li)
    data_lag1 = data_lag1.drop(['Wspd','Wdir','Etmp','Itmp','Ndir','Pab1','Pab2','Pab3','Prtv'],axis = 1)
    #     data_lag1

    #分割数据集
    train_feat, train_label, val_feat, val_label, test_feat, test_label = dataset_split(data_lag1)

    # 模型训练
    gbm1 = lgb_(train_feat, train_label, val_feat, val_label)
    joblib.dump(gbm1, './checkpoint/lgb.pkl')

    #获取修改值的index
    index_list = list(test_feat['index'])

    #构造单步预测的test集
    test_feat = lag_insert(test_feat, lag_n,index_list)
    # print(test_feat)

    #取出288时刻中的第一个时刻进行预测
    real_test_feat,real_test_label = get_test_feat(test_feat,test_label,34993)

    #保存信息的list
    pred_list,rmse_list, mae_list = [], [], []

    #第一个时刻进行预测，pred_1的长度为134，即134个风机的第一个时刻的预测
    gbm1 = joblib.load('./checkpoint/lgb.pkl')
    pred_1,pred_rmse,pred_mae = lgb_test_new(gbm1, data, real_test_feat, real_test_label)

    #记录预测以及误差
    pred_list.extend(pred_1)
    rmse_list.append(pred_rmse)
    mae_list.append(pred_mae)

    #循环预测
    print('开始循环预测')
    pred_list,rmse_list,mae_list = round_pred(gbm1,test_feat,test_label,pred_1,pred_list, rmse_list, mae_list,index_list)

    return pred_list,rmse_list,mae_list


def single_step(lag_n):
    # 做lag特征
    #     feat_columns = ['Wspd']
    #     li = [i for i in range(288,288+wspd_lag)]
    #     data_lag1 = lag_feature(data, feat_columns, li)

    feat_columns = ['Patv']
    li = [i for i in range(1,lag_n+1)]
    data_lag1 = lag_feature(data, feat_columns, li)
    data_lag1 = data_lag1.drop(['Wspd','Wdir','Etmp','Itmp','Ndir','Pab1','Pab2','Pab3','Prtv'],axis = 1)
    #     data_lag1

    #分割数据集
    train_feat, train_label, val_feat, val_label, test_feat, test_label, test_individual_feat, test_individual_label = dataset_split(data_lag1)

    # 模型训练
    gbm1 = lgb_(train_feat, train_label, val_feat, val_label)

    #获取修改值的index
    index_list = list(test_feat['index'])

    #构造单步预测的test集
    test_feat = lag_insert(test_feat, lag_n,index_list)
    print(test_feat)

    #取出288时刻中的第一个时刻进行预测
    real_test_feat,real_test_label = get_test_feat(test_feat,test_label,34993)

    #保存信息的list
    pred_list,rmse_list, mae_list = [], [], []

    #第一个时刻进行预测，pred_1的长度为134，即134个风机的第一个时刻的预测
    pred_1,pred_rmse,pred_mae = lgb_test_new(gbm1, data, real_test_feat, real_test_label)

    #记录预测以及误差
    pred_list.extend(pred_1)
    rmse_list.append(pred_rmse)
    mae_list.append(pred_mae)

    #循环预测
    # print('开始循环预测')
    pred_list,rmse_list,mae_list = round_pred(gbm1,test_feat,test_label,pred_1,pred_list, rmse_list, mae_list,index_list)

    return pred_list,rmse_list,mae_list


def forecast(settings):
    data = pd.read_csv(settings['file'])
    
    checkpoints = settings["checkpoints"]
    start_col = settings["start_col"]
    day_len = settings["day_len"]
    capacity = settings["capacity"]
    pred_file = settings["pred_file"]
    is_debug = settings["is_debug"]
    lag_n = settings["lag_n"]
    
    data = data.reset_index()
    data['time_idx'] = data.groupby('TurbID')['index'].rank().astype(int)
    data = data.fillna(method='ffill')
    # data = opreate_abnormal(data)
    # print('ok')
    data = data[(data['Patv'].notnull()) & (data['Day']>150)]



    feat_columns = ['Wspd', 'Wdir', 'Prtv']
    li = [i for i in range(288, 288 + lag_n)]
    data_lag1 = lag_feature(data, feat_columns, li)

    data_lag1 = data_lag1[data_lag1['Prtv_lag_{}'.format(288 + lag_n - 1)].notnull()]

    test_data = data_lag1[data_lag1['Day'].isin([244, 245])]
    real_test_data_patv = test_data.Patv.copy()
    test_data.Patv = np.nan

    train_data = data_lag1[data_lag1['Day'] < 244]
    data_lag1 = pd.concat([train_data, test_data], axis=0)

    feat_columns = ['Patv']
    li = [i for i in range(1, lag_n + 1)]
    data_lag1 = lag_feature(data_lag1, feat_columns, li)

    data_lag1 = data_lag1.drop(['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv'], axis=1)

    val_data = data_lag1[data_lag1['Day'].isin([243, 242])]
    test_data = data_lag1[data_lag1['Day'].isin([244, 245])]
    train_data = data_lag1[data_lag1['Day'] < 242]

    test_data.Patv = real_test_data_patv.copy()

    test_data = test_data.reset_index(drop=True)

    test_data['time_idx'] = test_data.reset_index().groupby('TurbID')['index'].rank()

    insert_label = ["Patv_lag_" + str(i + 1) for i in range(lag_n)]
    insert_label = insert_label[::-1]

    start_index = search_timeidx_index(test_data)
    use_feat = list(['TurbID'] + list(test_data.columns[start_index:-1]))

    unchange_data = [i for i in use_feat if i not in insert_label]

    train_data = train_data[train_data['Patv_lag_{}'.format(lag_n)].notnull()]  # [use_feat]

    gbm = LGBMRegressor(objective='regression',
                        num_leaves=128,
                        learning_rate=0.07,
                        n_estimators=1000,
                        max_depth=7)


    gbm.fit(train_data[unchange_data + insert_label], train_data['Patv'].values,
            eval_set=[(val_data[unchange_data + insert_label], val_data['Patv'].values)],
            eval_metric='rmse', early_stopping_rounds=20)
    joblib.dump(gbm, './checkpoint/lgb.pkl')

    lag_queue_ID = {}
    for i in test_data.TurbID.unique():
        lag_queue_ID[i] = []
        for value in test_data[(test_data['time_idx'] == 1) & (test_data['TurbID'] == i)][insert_label].values[0]:
            lag_queue_ID[i].append(value)

    gbm = joblib.load('./checkpoint/lgb.pkl')
    predict_final = []
    for time_idx in range(1, 289):
        print(time_idx)
        if time_idx == 1:
            predict = gbm.predict(test_data[test_data['time_idx'] == time_idx][unchange_data + insert_label])
            predict_final.append(predict)
            for i in test_data.TurbID.unique():
                lag_queue_ID[i].append(predict[i - 1])
        else:
            lag_feature_list = []
            for i in test_data.TurbID.unique():
                lag_feature_list.append(lag_queue_ID[i][-lag_n:])
            lag_df = pd.DataFrame(lag_feature_list, columns=insert_label)
            new_data = pd.concat([test_data[test_data['time_idx'] == time_idx][unchange_data].reset_index(drop=True),
                                  lag_df], axis=1)
            predict = gbm.predict(new_data)
            predict_final.append(predict)
            for i in test_data.TurbID.unique():
                lag_queue_ID[i].append(predict[i - 1])

    pred_list_134 = []
    for i in range(len(predict_final[0])):
        id_temp_pred = []
        for j in range(len(predict_final)):
            id_temp_pred.append(predict_final[j][i])
        pred_list_134.append(id_temp_pred)


    # data_patv = list(real_test_data_patv)
    test_patv_label = []
    for i in range(1, 135):
        temp_list1 = test_data[test_data['TurbID'] == i]
        test_patv_label.append(list(temp_list1.Patv))

    # rmse_list = []
    # mae_list = []
    # for i in range(1, 135):
    #     pred_rmse = mean_squared_error(test_patv_label[i - 1], pred_list_134[i - 1]) ** 0.5
    #     pred_mae = mean_absolute_error(test_patv_label[i - 1], pred_list_134[i - 1])
    #     rmse_list.append(pred_rmse)
    #     mae_list.append(pred_mae)
    return pred_list_134, test_patv_label, test_data
    # return rmse_list,mae_list



if __name__ == '__main__':
    env = prep_env()
    lag_n = 2
    data = pd.read_csv('wtbdata_245days.csv')
    pred_list_134, test_patv_label, test_data  = forcast(data,lag_n)
    rmse_list = []
    mae_list = []
    for i in range(1, 135):
        pred_rmse = mean_squared_error(test_patv_label[i - 1], pred_list_134[i - 1]) ** 0.5
        pred_mae = mean_absolute_error(test_patv_label[i - 1], pred_list_134[i - 1])
        rmse_list.append(pred_rmse)
        mae_list.append(pred_mae)

    # print('rmse:',np.mean(rmse_list))
    # print('mae:',np.mean(mae_list))

    score = 0.5*(np.sum(mae_list)+np.sum(rmse_list))/1000
