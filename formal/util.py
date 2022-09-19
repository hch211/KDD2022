import os
import gc
import time
import math
import datetime
from math import log, floor
from sklearn.neighbors import KDTree
import copy

import numpy as np
import modin.pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm as tqdm
from scipy.signal import find_peaks

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import pywt
from statsmodels.robust import mad

import scipy
import statsmodels
from scipy import signal
import statsmodels.api as sm
from fbprophet import Prophet
from scipy.signal import butter, deconvolve
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.dataframe_functions import roll_time_series

from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, GroupKFold

import warnings
warnings.filterwarnings("ignore")

# 用lgb测试只需要data的方法，打印出分数
# li = ['wspd_clip', 'location_id']
def test_data_func(data, func_li):
    raw_data = data
    for func in func_li:
        data = eval(func)(raw_data)
        data = remove_abnormal(data)
        data = data.drop('Tmstamp',axis=1)
        train_feat, train_label, val_feat, val_label, test_feat, test_label, test_individual_feat, test_individual_label = dataset_split(data)
        gbm = lgb_(train_feat, train_label, val_feat, val_label)
        print(func+' performance:')
        lgb_test(gbm, data, test_feat, test_label)
        
def test_data_fc_func(data, feat_columns, func_li):
    
    raw_data = data
    for func in func_li:
        data = eval(func)(raw_data, feat_columns)
        data = remove_abnormal(data)
        data = data.drop('Tmstamp',axis=1)
        train_feat, train_label, val_feat, val_label, test_feat, test_label, test_individual_feat, test_individual_label = dataset_split(data)
        gbm = lgb_(train_feat, train_label, val_feat, val_label)
        print(func+' performance:')
        lgb_test(gbm, data, test_feat, test_label)
        
        
def test_data_fc_li_func(data, feat_columns, li, func_li):
    
    raw_data = data
    for func in func_li:
        data = eval(func)(raw_data, feat_columns, li)
        data = remove_abnormal(data)
        data = data.drop('Tmstamp',axis=1)
        train_feat, train_label, val_feat, val_label, test_feat, test_label, test_individual_feat, test_individual_label = dataset_split(data)
        gbm = lgb_(train_feat, train_label, val_feat, val_label)
        print(func+' performance:')
        lgb_test(gbm, data, test_feat, test_label)


# 这个li是[[],[]]
def test_lag(data, feat_columns, li):
    
    raw_data = data
    
    best_rmse = float('inf')
    best_mae = float('inf')
    for l in li:
        data = lag_feature(raw_data, feat_columns, l)
        data = remove_abnormal(data)
        data = data.drop('Tmstamp',axis=1)
        train_feat, train_label, val_feat, val_label, test_feat, test_label, test_individual_feat, test_individual_label = dataset_split(data)
        gbm = lgb_(train_feat, train_label, val_feat, val_label)

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
            if not ((test_label.values[i] <= 0 and test_feat.Wspd.values[i] > 2.5) or test_feat.Pab1.values[i] > 89 or test_feat.Pab2.values[i] > 89 or test_feat.Pab3.values[i] > 89):
                y_pred_clear.append(y_pred[i])
                test_label_clear.append(test_label.values[i])

        # 模型评估
    #     print('The rmse of prediction is:', mean_squared_error(test_label, y_pred) ** 0.5)
        rmse = mean_squared_error(test_label_clear, y_pred_clear) ** 0.5
        mae = mean_absolute_error(test_label_clear, y_pred_clear)
        print('The rmse of prediction is:', rmse)

        # 模型评估
        print('The mae of prediction is:', mae)
        
        if best_rmse > rmse:
            best_rmse = rmse
            best_li = l
        if best_mae > mae:
            best_mae = mae
            
        
            
    print('best rmse:', best_rmse)
    print('best mae:', best_mae)
    
    return best_li
    
        
        
        
        
        
def wspd_clip(data):
    for index, row in data.iterrows():
        if row.Wspd > 12:
            data.Wspd[index] = 12
        elif row.Wspd < 3:
            data.Wspd[index] = 3
    
    return data

def location_id(data, location_data):
    x = list(location_data['x'])
    y = list(location_data['y'])
    z = list(location_data['TurbID'])
    data_0,data_1000,data_2000,data_3000,data_4000,data_5000 = [], [], [], [], [], []
    for i in range(len(x)):
        if x[i]<500:
            data_0.append(z[i])
        elif 500<x[i]<1500:
            data_1000.append(z[i])
        elif 1500<x[i]<2500:
            data_2000.append(z[i])
        elif 2500<x[i]<3700:
            data_3000.append(z[i])
        elif 3700<x[i]<4700:
            data_4000.append(z[i])
        else:
            data_5000.append(z[i])
    location_list = []
    for ID in data['TurbID']:
        
        if ID in data_0:
            location_list.append(0)

        elif ID in data_1000:
            location_list.append(1)

        elif ID in data_2000:
            location_list.append(2)
            
        elif ID in data_3000:
            location_list.append(3)
            
        elif ID in data_4000:
            location_list.append(4)

        elif ID in data_5000:
            location_list.append(5)
    data['location_id'] = location_list
    
    return data

def get_time_week(data):
    tmp = []
        
    for index, row in data.iterrows():
        tmp.append(int((row.Day-1)*96+int(row.Tmstamp[:2])*4+int(row.Tmstamp[3:])/15))
        
    data['time_week'] = tmp
    
    return data

def get_time_day(data):
    tmp = []
    
    for index, row in data.iterrows():
        tmp.append(int(int(row.Tmstamp[:2])*4+int(row.Tmstamp[3:])/15))
    data['time_day'] = tmp
    
    return data

def get_time_hour(data):
    tmp = []

    for index, row in data.iterrows():
        tmp.append(int(row.Tmstamp[:2]))
    data['time_hour'] = tmp
    
    return data

def time_triangle(data):
    time = []
    for i in range(len(data.Tmstamp)):
        time.append(datetime.datetime.strptime(data.Tmstamp[i],'%M:%S'))
    data['timestamp'] = time
    timestamp_s = data['timestamp'].map(datetime.datetime.timestamp)
    day = 24*60*60
    data['Day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    data['Day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    
    return data

def Group_Normalization(data, feat_columns):
    scaler = lambda x : (x-np.min(x)) / (np.max(x)-np.min(x))
    temp_data = data.groupby('TurbID')[feat_columns].apply(scaler)
    temp_data.columns=[i+"_group_norm" for i in feat_columns]
    data = pd.concat([data,temp_data],axis=1)
    return data

def tmp_anomaly_replace(data):
    data['Etmp'] = data['Etmp'].replace(-272,np.nan)
    data['Etmp'] = data['Etmp'].fillna(method='ffill')
    data['Itmp'] = data['Itmp'].replace(-273.2,np.nan)
    data['Itmp'] = data['Itmp'].fillna(method='ffill')
    
    return data

def Etmp_anomaly_process(data):
    normal_etmp = data[(data["Etmp"] >= -50) & (data["Etmp"] <= 50)]["Etmp"]

    # 定义3σ原则表达式
    min_mask = data["Etmp"] < (normal_etmp.mean() - 3 * normal_etmp.std())
    max_mask = data["Etmp"] > (normal_etmp.mean() + 3 * normal_etmp.std())

    # 只要满足上诉表达式的任一个就为异常值，所以这里使用位与运算
    mask = min_mask | max_mask

    data.loc[mask,"Etmp"] = normal_etmp.mean()
    
    return data

def Itmp_anomaly_process(data):
    normal_itmp = data[(data["Itmp"] > 0) & (data["Itmp"] <= 50)]["Itmp"]

    # 定义3σ原则表达式
    min_mask = data["Itmp"] < (normal_itmp.mean() - 3 * normal_itmp.std())
    max_mask = data["Itmp"] > (normal_itmp.mean() + 3 * normal_itmp.std())

    # 只要满足上诉表达式的任一个就为异常值，所以这里使用位与运算
    mask = min_mask | max_mask

    data.loc[mask,"Itmp"] = normal_itmp.mean()
    
    return data

def raw_data_peak(data, li, distance=1, show=False):
    peak_list = []
    for i in range(len(li)):
        plt.figure(figsize=(20,5))
        x = data[li[i]].values
        peaks, _ = find_peaks(x, height=0, distance=distance)
        if show == True:
            plt.plot(x)
            plt.plot(peaks, x[peaks], "x")
            plt.plot(np.zeros_like(x), "--", color="gray")
            plt.title(li[i])
            plt.show()
#         peak_list.append(np.diff(peaks).tolist())
        tmp_peak_list = []
        for j in range(len(data[li[i]])):
            if j in np.diff(peaks):
                tmp_peak_list.append(1)
            else:
                tmp_peak_list.append(0)
        data[li[i]+'_peak'] = tmp_peak_list
    return data

# 输入预测值序列
def pred_peak(data, distance=1):
    plt.figure(figsize=(20,5))
    x = data
    peaks, _ = find_peaks(x, height=0, distance=distance)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()
    return np.diff(peaks).tolist()

def lag_feature(data, feat_columns, li):    
    # 在对应的位置设置lag列
    for shift_num in li:
        temp_data = data.groupby(['TurbID'])[feat_columns].shift(shift_num)
        temp_data.columns=[i+"_lag_"+str(shift_num) for i in feat_columns]
        data = pd.concat([data,temp_data],axis=1)
    return data 

def unique_avg_prtv(data, feat_columns):
    for feat in feat_columns:
        temp_data = data.groupby(feat)[['Prtv']].transform('mean')##这里还可以有max,min之类的操作
        temp_data.columns=[feat+"_avg_Prtv"]  
        data = pd.concat([data,temp_data],axis=1)
    return data

def unique_avg_patv(data, feat_columns):
    for feat in feat_columns:
        temp_data = data.groupby(feat)[['Patv']].transform('mean')##这里还可以有max,min之类的操作
        temp_data.columns=[feat+"_avg_Patv"]
        data = pd.concat([data,temp_data],axis=1)
    return data

def self_minus(data, feat_columns, li):
    for diff_num in li:
        temp_data = data.groupby('TurbID')[feat_columns].diff(diff_num)
        temp_data.columns=[i+"_diff_"+str(diff_num) for i in feat_columns]
        data = pd.concat([data,temp_data],axis=1)
    return data

def group(data, far):
    group_row = []
    for i, row in data.iterrows():
        if row['TurbID'] in far:
            group_row.append(0)
        else:
            group_row.append(1)
    data['group'] = group_row
    return data

def no_change(data, feat_columns):
    """统计某列多少个时间点没有变化"""
    
    temp_data = data.groupby('TurbID')[feat_columns].diff(1)
    temp_data.columns=[i+"_diff_1" for i in feat_columns]
    data = pd.concat([data,temp_data],axis=1)
    
    for i in feat_columns:
        data["no_change_"+i] = data.index.tolist()

    for i in feat_columns:
        temp = 0
        for idx, row in data.iterrows():
            if row[i+"_diff_1"] == 0:
                temp += 1
            else:
                temp = 0
            data["no_change_"+i][idx] = temp
    
    data = data.drop([i+"_diff_1" for i in feat_columns],axis=1)
    
    return data

def raw_extract(data, cid, csort):
    if 'Tmstamp' in data.columns.tolist():
        data = data.drop('Tmstamp',axis=1)
    extracted_features = extract_features(data, column_id=cid, column_sort=csort)
    return pd.merge(data, extracted_features, left_on='TurbID', right_index=True, how='left', sort=True)

# 暂时先用每台机器均值
def filtered_extract(data, cid, csort):
    if 'Tmstamp' in data.columns.tolist():
        data = data.drop('Tmstamp',axis=1)
    extracted_feat = extract_features(data, column_id=cid, column_sort=csort)
    impute(extracted_feat)
    if 'TurbID_avg_patv' not in data.columns.tolist():
        unique_avg(data, ['TurbID'])
    features_filtered = select_features(extracted_feat, data['TurbID_avg_patv'].unique())
    return pd.merge(data, features_filtered, left_on='TurbID', right_index=True, how='left', sort=True)
    
# roll feature to be done
def roll_extract(data, cid, csort):
    if 'Tmstamp' in data.columns.tolist():
        data = data.drop('Tmstamp',axis=1)
    df_rolled = roll_time_series(data, column_id=cid, column_sort=csort)
    df_features = extract_features(df_rolled, column_id=cid, column_sort=csort)
    return pd.merge(data, df_features, left_on='TurbID', right_index=True, how='left', sort=True)

def rolling_mean(data, feat_columns,li):
    for roll_num in li:
        temp_data = data.groupby('TurbID')[feat_columns].rolling(roll_num).mean().reset_index(drop=True)
        temp_data.columns=[i+"_roll_"+str(roll_num)+'_mean' for i in feat_columns]
        data = pd.concat([data,temp_data],axis=1)
    return data

def rolling_max(data, feat_columns,li):
    for roll_num in li:
        temp_data = data.groupby('TurbID')[feat_columns].rolling(roll_num).max().reset_index(drop=True)
        temp_data.columns=[i+"_roll_"+str(roll_num)+'_max' for i in feat_columns]
        data = pd.concat([data,temp_data],axis=1)
    return data

def rolling_min(data, feat_columns,li):
    for roll_num in li:
        temp_data = data.groupby('TurbID')[feat_columns].rolling(roll_num).min().reset_index(drop=True)
        temp_data.columns=[i+"_roll_"+str(roll_num)+'_min' for i in feat_columns]
        data = pd.concat([data,temp_data],axis=1)
    return data

def dataset_split(data):
    if 'Tmstamp' in data.columns.tolist():
        data = data.drop('Tmstamp',axis=1)
    train_data = data[lambda x: x.time_idx < 30816]
    val_data = data[(data.time_idx>=30816) & (data.time_idx<34993)]
    test_data = data[lambda x: x.time_idx >= 34993]
#     test_data = data.groupby('TurbID').tail(data.Tmstamp.nunique()*2)
#     train_data = data.drop(data.groupby('TurbID').tail(data.Tmstamp.nunique()*2).index)
#     val_data = train_data.groupby('TurbID').tail(train_data.Tmstamp.nunique()*2)
#     train_data = train_data.drop(train_data.groupby('TurbID').tail(train_data.Tmstamp.nunique()*2).index)  


#     train_data = data[lambda x: x.time_idx < 22032]
#     val_data = data[(data.time_idx>=22032) & (data.time_idx < 24336)]
#     test_data = data[lambda x: x.time_idx >= 24336]


    train_feat = train_data.drop('Patv',axis=1)
    train_label = train_data['Patv']
    val_feat = val_data.drop('Patv',axis=1)
    val_label = val_data['Patv']
    test_feat = test_data.drop('Patv',axis=1)
    test_label = test_data['Patv']
    test_individual_feat = []
    test_individual_label = []
    for i in range(data.TurbID.nunique()):
        tmp = test_data.drop('Patv',axis=1).loc[data['TurbID'] == i+1]
        tmp.columns = list(range(len(tmp.columns.tolist())))
        test_individual_feat.append(tmp)
        test_individual_label.append(test_data['Patv'].loc[data['TurbID'] == i+1])
    
    # 为了处理tsfresh的特殊符号列名，直接将所有列名写成数字
#     train_feat.columns = list(range(len(train_feat.columns.tolist())))
#     test_feat.columns = list(range(len(test_feat.columns.tolist())))
    
    return train_feat, train_label, val_feat, val_label, test_feat, test_label, test_individual_feat, test_individual_label

def plot_data_TurbID(data, column, li):
    TurbIDs = sorted(list(set(data['TurbID'])))
    d_cols = [column]
    for j in range(data.Day.nunique()):
        plt.figure(figsize=(20,5))
        for i in range(len(li)):
            turb_data = data.loc[data['Day'] == j+1]
            turb_data = turb_data.loc[turb_data['TurbID'] == TurbIDs[li[i]]].set_index('TurbID')[d_cols]
            turb_data = turb_data.T.values[0]
            plt.plot(range(len(turb_data)), turb_data)
            plt.title('Day:{}'.format(j))
        plt.show()
        
def plot_data_time(data, column, li, time):
    TurbIDs = sorted(list(set(data['TurbID'])))
    for d in column:
        d_col = [d]
        plt.figure(figsize=(20,5))
        for i in range(len(li)):
            turb_data = data.loc[data['time_idx'].isin(time)]
            turb_data = turb_data.loc[turb_data['TurbID'] == TurbIDs[li[i]]].set_index('TurbID')[d_col]
            turb_data = turb_data.T.values[0]
            plt.plot(range(len(turb_data)), turb_data)
            plt.title(d)
        plt.show()
        
def avg_compare_turb(feat, label, gbm):
    plt.figure(figsize=(20,5))
    x = []
    y = []
    y_ = []
    for i in range(len(feat)):
        pred_avg = np.mean(gbm.predict(feat[i]))
        label_avg = np.mean(label[i])
        x.append(i+1)
        y.append(pred_avg)
        y_.append(label_avg)
    sorted_id = sorted(range(len(y)), key=lambda k: y[k], reverse=False)
    sorted_id_str = []
    for i in range(len(sorted_id)):
        sorted_id_str.append(str(sorted_id[i]))
    zipped = zip(y,y_)
    sorted_zip = sorted(zipped, key=lambda x:x[0])
    result = zip(*sorted_zip)
    y, y_ = [list(x) for x in result]
    plt.scatter(sorted_id_str, y, label='pred')
    plt.scatter(sorted_id_str, y_, label='actual')
    plt.legend()
    plt.show()
    
    
    close = []
    far = []
    diff_mean = np.mean(np.absolute(np.diff([y, y_],axis=0)))
    for i in range(len(y)):
        if np.absolute(y[i] - y_[i]) < diff_mean:
            close.append(sorted_id[i])
        else:
            far.append(sorted_id[i])
    return close, far

def avg_compare_time(feat, label, gbm):
    plt.figure(figsize=(20,5))
    pred_avg = 0
    for d in feat:
        pred_avg += gbm.predict(d)
    pred_avg /= len(feat)
    
    for i in range(len(label)):
        if i == 0:
            label_avg = label[i].tolist()
        else:
            label_avg = np.sum([label_avg, label[i].tolist()],axis=0)
    for i in range(len(label_avg)):
        label_avg[i] /= len(feat)
        
    sorted_id = sorted(range(len(pred_avg)), key=lambda k: pred_avg[k], reverse=False)
    sorted_id_str = []
    for i in range(len(sorted_id)):
        sorted_id_str.append(str(sorted_id[i]))
        
    zipped = zip(pred_avg, label_avg)
    sorted_zip = sorted(zipped, key=lambda x:x[0])
    result = zip(*sorted_zip)
    pred_avg, label_avg = [list(x) for x in result]
    
    plt.scatter(sorted_id_str, pred_avg, label='pred')
    plt.scatter(sorted_id_str, label_avg, label='actual')
    plt.legend()
    plt.show()
    
    
    
def lgb_(train_feat, train_label, test_feat, test_label):
    gbm = LGBMRegressor(objective='regression', num_leaves=80, learning_rate=0.07, n_estimators=1000, max_depth=7)
    start = time.time()
    gbm.fit(train_feat, train_label, eval_set=[(test_feat, test_label)], eval_metric='rmse', early_stopping_rounds=20)
    end = time.time()
    print('training time: ', end-start)
    
    return gbm

def lgb_cv(train_feat, train_label, test_feat, test_label):
    params = {'num_leaves': 80, 
              'max_depth': 7,
              'learning_rate': 0.3,
              "boosting": "gbdt",
              "feature_fraction": 0.3,  
              "bagging_freq": 1,
              "bagging_fraction": 0.8,
              "bagging_seed": 1,
              "lambda_l1": 0.3,             #l1
              'lambda_l2': 0.01,     #l2
              "verbosity": -1,
              "nthread": -1,                
              'metric': 'rmse', 
              "random_state": 1, 
              'n_estimators': 1000,
              }

    test = test_feat.copy()

    NFOLD = 5

    groups = train_feat.TurbID.values
    group_kfold = GroupKFold(n_splits=5)
    
    cv_pred = 0
    valid_best = 0
    
    for train_index, test_index in group_kfold.split(train_feat, train_label, groups):
        X_train, X_validate = train_feat.iloc[train_index,:], train_feat.iloc[test_index,:]
        label_train, label_validate = train_label.iloc[train_index], train_label.iloc[test_index]
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

        gbm = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid,early_stopping_rounds=20)

        preds_last = gbm.predict(test, num_iteration=gbm.best_iteration)
        cv_pred += gbm.predict(test, num_iteration=gbm.best_iteration)
        valid_best += gbm.best_score['valid_0']['rmse']
        
    
    return gbm


def lgb_test(gbm, data, test_feat, test_label, show=False):
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
        if not ((test_label.values[i] <= 0 and test_feat.Wspd.values[i] > 2.5) or test_feat.Pab1.values[i] > 89 or test_feat.Pab2.values[i] > 89 or test_feat.Pab3.values[i] > 89):
            y_pred_clear.append(y_pred[i])
            test_label_clear.append(test_label.values[i])
    
    
    # 模型评估
#     print('The rmse of prediction is:', mean_squared_error(test_label, y_pred) ** 0.5)
    print('The rmse of prediction is:', mean_squared_error(test_label_clear, y_pred_clear) ** 0.5)

    # 模型评估
    print('The mae of prediction is:', mean_absolute_error(test_label_clear, y_pred_clear))

    # 特征重要度
    print('{}\nFeature importances:{}'.format([column for column in test_feat], list(gbm.feature_importances_)))
    
    if show == True:
        plt.figure(figsize=(20,5))
        plt.plot(range(len(y_pred_clear[:2000])), y_pred_clear[:2000], label='pred')
        plt.plot(range(len(test_label_clear[:2000])), test_label_clear[:2000], label='label')
        plt.legend()
    
    
# 填充所有nan为0
def fillna_0(data):
    data.replace(to_replace=np.nan, value=0, inplace=True)
    
    return data

# 在fillna_0之后，去除训练集中的所有异常值
def remove_abnormal(data):
    data.replace(to_replace=np.nan, value=0, inplace=True)
    data = data.drop(data[(((data.Patv <= 0) & (data.Wspd > 2.5)) | (data.Pab1 > 89) | (data.Pab2 > 89) | (data.Pab3 > 89))].index)
    data = data.drop(data[(((data.Ndir < -720) | (data.Ndir > 720)) | ((data.Wdir > 180) | (data.Wdir < -180)))].index)
    
    return data