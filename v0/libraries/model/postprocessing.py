# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np
from scipy.ndimage import convolve
from scipy.signal import medfilt
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
