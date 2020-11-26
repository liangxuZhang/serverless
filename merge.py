# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:19:35 2020

@author: zlx
"""

import pandas as pd

cpu = pd.read_csv('baseline1.csv')
job = pd.read_csv('test1.csv')
to_drop_cols = [col for col in cpu.columns if col.endswith('_JOB_NUMS') ] + ['ID']
cpu.drop(to_drop_cols, axis=1, inplace=True)
for col in [f'CPU_USAGE_{i}' for i in range(1,6)]:
    job[col] = cpu[col]

    
job = job[['ID',
           'CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1', 
           'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2', 
           'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3', 
           'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4', 
           'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5']]

job.to_csv('baseline2.csv',index = False)