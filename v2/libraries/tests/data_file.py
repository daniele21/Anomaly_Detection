#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:54:32 2020

@author: daniele
"""

#%%
import pandas as pd
import cv2
from libraries.utils import Paths
from libraries.model.dataset import computeMask
from copy import deepcopy
from matplotlib import pyplot as plt

paths = Paths()

def applyMask(image, mask):
    
    masked_image = deepcopy(image)    
    masked_image[mask==1, 0] = 255
    
    return masked_image
#%%
path_file = './libraries/dataset_package/'
namefile = 'train_unique.csv'

#%%

data = pd.read_csv(path_file + namefile, index_col=0)
data.head()

#data_unique = pd.read_csv(path_file + 'train_unique.csv')
#%%
data1 = data.loc[data.Class_Id == '1']
data2 = data.loc[data.Class_Id == '2']
data3 = data.loc[data.Class_Id == '3']
data4 = data.loc[data.Class_Id == '4']

#%%
data1.to_csv(path_file + 'data_1_unique.csv')
data2.to_csv(path_file + 'data_2_unique.csv')
data3.to_csv(path_file + 'data_3_unique.csv')
data4.to_csv(path_file + 'data_4_unique.csv')

#%%
data_list = [data1, data2, data3, data4]
dest_path = paths.images_defects
defect = 1
i = 0

for i_data in data_list:
    i = 0
    for row in range(100):
        index_file = i_data['index'].iloc[i]
        print('Imagen°{}: im_{}'.format(i, index_file))
        
        filename    = i_data.iloc[row].Image_Id
        enc_pixels  = i_data.iloc[row].Encoded_Pixels

        image = cv2.imread(paths.images_path + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = computeMask(enc_pixels, image)  
        masked_image = applyMask(image, mask)
        
        cv2.imwrite(dest_path + str(defect) + '/' + '{}_{}.png'.format(index_file, filename), image)
        cv2.imwrite(dest_path + str(defect) + '/' + '{}_{}_MASK.png'.format(index_file, filename), masked_image)
        
        i += 1
        
    plt.imshow(masked_image)

#        break
    defect += 1
#    break
        
