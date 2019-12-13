# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np
from scipy.ndimage import convolve
#%% CONVOLUTION

anom = np.array([[0, 1.5, 2.3],
                 [1, 1.1, 2.5]])
anom

kernel = np.ones(len(anom))
kernel
kernel /= len(kernel)
kernel 

convolve(anom, kernel)
