# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from libraries.dataset_package.dataset_manager import computeMask, Image

from scipy.ndimage import convolve, median_filter
from scipy.signal import medfilt
import cv2
#%% CONVOLUTION

anom = np.array([[0, 1.5, 2.3],
                 [1, 1.1, 2.5]])
anom

kernel = np.ones((2,3))
kernel
kernel /= kernel.shape[0]*kernel.shape[1]
kernel 

convolve(anom, kernel)
medfilt(anom)
#median_filter(anom, kernel)

#%%
train = pd.read_csv('libraries/dataset_package/train_unique.csv', index_col=0)
i = 0
#%%
i += 1
#%%
enc = train.iloc[i].Encoded_Pixels
filename = train.iloc[i].Image_Id

img = Image(filename)
mask = computeMask(enc, img)

plt.imshow(mask)

#%%
#k = np.ones((256,1600), dtype='uint8')
med = median_filter(mask, size=(32,4))
plt.show(med)
plt.show()

kernel = np.ones((256,1600)) / (256*1600) 
conv = convolve(mask, kernel)
plt.show(conv)
plt.show()
#%%
med = cv2.medianBlur(mask, ksize=21)
print(len(np.where(med)[0]))
print(len(np.where(mask)[0]))

gauss = cv2.GaussianBlur(med, ksize=7)
print(len(np.where(gauss)[0]))

plt.imshow(med)
plt.show()
plt.imshow(mask)
plt.show()
plt.imshow(gauss)
plt.show()

