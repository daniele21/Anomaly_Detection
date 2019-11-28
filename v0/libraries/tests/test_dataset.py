#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:20:04 2019

@author: daniele
"""
#%% IMPORTS
import cv2
from matplotlib import pyplot as plt

import torch

from libraries.model.dataset import generateDataloader, getCifar10, newLoadData
from libraries.model.dataset import collectAnomalySamples
from libraries.model.options import Options
from libraries.utils import Paths

paths = Paths()
#%%

opt = Options()

#%%

cifar_dataloader = getCifar10(opt)

#%%

steel_dataloader = generateDataloader(opt)

#%%

image = cv2.imread(paths.normal_images_path + '0/0002cc93b.jpg_(x,y) -> (32,72)')
image.shape
plt.imshow(image)

#%%
image = image.flatten()
image.shape
image = image.reshape(32,32,3)
image.shape
image = image.transpose(1,0,2)
image.shape

plt.imshow(image)
#%%

tensor_image = torch.Tensor(image).cuda()
tensor_image.shape

#%%
opt = Options()
opt.patch_per_im = 2000

a = newLoadData(opt)
#%%

c = np.where(a[3])
c[0].shape
c

#%%

anom = collectAnomalySamples(2)

