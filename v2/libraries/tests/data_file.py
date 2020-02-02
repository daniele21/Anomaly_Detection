#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:54:32 2020

@author: daniele
"""

#%%
import pandas as pd

#%%
path_file = './libraries/dataset_package/'
namefile = 'train_db.csv'

#%%

data = pd.read_csv(path_file + namefile, index_col=0)
data.head()

#data_unique = pd.read_csv(path_file + 'train_unique.csv')

data1 = data.loc[data.Class_Id == 1]
data2 = data.loc[data.Class_Id == 2]
data3 = data.loc[data.Class_Id == 3]
data4 = data.loc[data.Class_Id == 4]

data1.to_csv(path_file + 'data_1.csv')
data2.to_csv(path_file + 'data_2.csv')
data3.to_csv(path_file + 'data_3.csv')
data4.to_csv(path_file + 'data_4.csv')

