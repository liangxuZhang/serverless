# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:33:11 2020

@author: zlx
"""

import warnings
warnings.simplefilter('ignore')

import gc

import numpy as np
import pandas as pd
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb


train = pd.read_csv('train.csv')
train = train.sort_values(by=['QUEUE_ID', 'DOTTING_TIME']).reset_index(drop=True)

test = pd.read_csv('evaluation_public.csv')
test = test.sort_values(by=['ID', 'DOTTING_TIME']).reset_index(drop=True)

sub_sample = pd.read_csv('submit_example.csv')

del train['STATUS']
del train['PLATFORM']
del train['RESOURCE_TYPE']

del test['STATUS']
del test['PLATFORM']
del test['RESOURCE_TYPE']

# 时间排序好后也没什么用了

del train['DOTTING_TIME']
del test['DOTTING_TIME']

le = LabelEncoder()
train['QUEUE_TYPE'] = le.fit_transform(train['QUEUE_TYPE'].astype(str))
test['QUEUE_TYPE'] = le.transform(test['QUEUE_TYPE'].astype(str))



# 只用 CPU_USAGE 和 MEM_USAGE
to_drop_cols = [col for col in train.columns if col.endswith('_JOB_NUMS')] + ['DISK_USAGE']

train.drop(to_drop_cols, axis=1, inplace=True)
test.drop(to_drop_cols, axis=1, inplace=True)

# t0 t1 t2 t3 t4  ->  t5 t6 t7 t8 t9 
# t1 t2 t3 t4 t5  ->  t6 t7 t8 t9 t10

df_train = pd.DataFrame()

for id_ in tqdm(train.QUEUE_ID.unique()):
    df_tmp = train[train.QUEUE_ID == id_]
    features = list()
    t_cpu = list()
    values = df_tmp.values
    for i, _ in enumerate(values):
        if i + 10 < len(values):
            li_v = list()
            li_v.append(values[i][0])
            li_cpu = list()
            for j in range(5):
                li_v.extend(values[i+j][3:].tolist())
                li_cpu.append(values[i+j+5][3])
            features.append(li_v)
            t_cpu.append(li_cpu)
    df_feat = pd.DataFrame(features)
    df_feat.columns = ['QUEUE_ID', 
                       'CPU_USAGE_1', 'MEM_USAGE_1', 
                       'CPU_USAGE_2', 'MEM_USAGE_2', 
                       'CPU_USAGE_3', 'MEM_USAGE_3', 
                       'CPU_USAGE_4', 'MEM_USAGE_4', 
                       'CPU_USAGE_5', 'MEM_USAGE_5', 
                      ]
    df_cpu = pd.DataFrame(t_cpu)
    df_cpu.columns = ['cpu_1', 'cpu_2', 'cpu_3', 'cpu_4', 'cpu_5']
    df = pd.concat([df_feat, df_cpu], axis=1)
  #  print(f'QUEUE_ID: {id_}, lines: {df.shape[0]}')
    df_train = df_train.append(df)
df_test = pd.DataFrame()

for id_ in tqdm(test.QUEUE_ID.unique()):
    df_tmp = test[test.QUEUE_ID == id_]
    features = list()
    values = df_tmp.values
    for i, _ in enumerate(values):
        if i % 5 == 0:
            li_v = list()
            li_v.append(values[i][0])
            li_v.append(values[i][1])
            for j in range(5):
                li_v.extend(values[i+j][4:].tolist())
            features.append(li_v)
    df_feat = pd.DataFrame(features)
    df_feat.columns = ['ID', 'QUEUE_ID', 
                       'CPU_USAGE_1', 'MEM_USAGE_1', 
                       'CPU_USAGE_2', 'MEM_USAGE_2', 
                       'CPU_USAGE_3', 'MEM_USAGE_3', 
                       'CPU_USAGE_4', 'MEM_USAGE_4', 
                       'CPU_USAGE_5', 'MEM_USAGE_5', 
                      ]
    df = df_feat.copy()
   # print(f'QUEUE_ID: {id_}, lines: {df.shape[0]}')
    df_test = df_test.append(df)


# 行内统计特征

df_train['cpu_mean'] = df_train[[f'CPU_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_train['cpu_std'] = df_train[[f'CPU_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_train['cpu_diff'] = df_train['CPU_USAGE_5'] - df_train['CPU_USAGE_1']
df_train['cpu_max'] = df_train[[f'CPU_USAGE_{i}' for i in range(1,6)]].max(axis=1)
df_train['mem_mean'] = df_train[[f'MEM_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_train['mem_std'] = df_train[[f'MEM_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_train['mem_max'] = df_train[[f'MEM_USAGE_{i}' for i in range(1,6)]].max(axis=1)

df_test['cpu_mean'] = df_test[[f'CPU_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_test['cpu_std'] = df_test[[f'CPU_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_test['cpu_diff'] = df_test['CPU_USAGE_5'] - df_test['CPU_USAGE_1']
df_test['cpu_max'] = df_test[[f'CPU_USAGE_{i}' for i in range(1,6)]].max(axis=1)
df_test['mem_mean'] = df_test[[f'MEM_USAGE_{i}' for i in range(1,6)]].mean(axis=1)
df_test['mem_std'] = df_test[[f'MEM_USAGE_{i}' for i in range(1,6)]].std(axis=1)
df_test['mem_max'] = df_test[[f'MEM_USAGE_{i}' for i in range(1,6)]].max(axis=1)



def run_lgb_qid(df_train, df_test, target, qid):
    
    feature_names = list(
        filter(lambda x: x not in ['QUEUE_ID', 'CU', 'QUEUE_TYPE'] + [f'cpu_{i}' for i in range(1,6)], 
               df_train.columns))
    
    # 提取 QUEUE_ID 对应的数据集
    df_train = df_train[df_train.QUEUE_ID == qid]
    df_test = df_test[df_test.QUEUE_ID == qid]
    
    #print(f"QUEUE_ID:{qid}, target:{target}, train:{len(df_train)}, test:{len(df_test)}")
    
    model = lgb.LGBMRegressor(num_leaves=64,
                              max_depth=8,
                              learning_rate=0.06,
                              n_estimators=10000,
                              subsample=0.9,
                              feature_fraction=0.8,
                              reg_alpha=0.4,
                              reg_lambda=0.8,
                              random_state=2000)
    oof = []
    prediction = df_test[['ID', 'QUEUE_ID']]
    prediction[target] = 0
    
    kfold = KFold(n_splits=10, random_state=2000)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]
        
        lgb_model = model.fit(X_train, 
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                              eval_metric='mse',
                              early_stopping_rounds=20)
        
        pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)
        df_oof = df_train.iloc[val_idx][[target, 'QUEUE_ID']].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)
        
        pred_test = lgb_model.predict(df_test[feature_names], num_iteration=lgb_model.best_iteration_)
        prediction[target] += pred_test / kfold.n_splits
        
        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()
        
    df_oof = pd.concat(oof)
    score = mean_squared_error(df_oof[target], df_oof['pred'])
    print('MSE:', score)

    return prediction, score



predictions = list()
scores = list()

for qid in tqdm(test.QUEUE_ID.unique()):    
    df = pd.DataFrame()
    for t in [f'cpu_{i}' for i in range(1,6)]:
        prediction, score = run_lgb_qid(df_train, df_test, target=t, qid=qid)
        if t == 'cpu_1':
            df = prediction.copy()
        else:
            df = pd.merge(df, prediction, on=['ID', 'QUEUE_ID'], how='left')            
        scores.append(score)

    predictions.append(df)



sub = pd.concat(predictions)

sub = sub.sort_values(by='ID').reset_index(drop=True)
sub.drop(['QUEUE_ID'], axis=1, inplace=True)
sub.columns = ['ID'] + [f'CPU_USAGE_{i}' for i in range(1,6)]

# 全置 0 都比训练出来的结果好
for col in [f'LAUNCHING_JOB_NUMS_{i}' for i in range(1,6)]:
    sub[col] = 0
    
sub = sub[['ID',
           'CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1', 
           'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2', 
           'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3', 
           'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4', 
           'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5']]

print(sub.shape)
sub.head()

sub['ID'] = sub['ID'].astype(int)

for col in [i for i in sub.columns if i != 'ID']:
    sub[col] = sub[col].apply(np.floor)
    sub[col] = sub[col].apply(lambda x: 0 if x<0 else x)
    sub[col] = sub[col].astype(int)
    
sub.head(10)

sub.to_csv('result1.csv', index=False)