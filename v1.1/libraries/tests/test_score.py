# -*- coding: utf-8 -*-

#%%
from libraries.model.score import nextPatch, anomalyScoreFromImage, anomalyScoreFromDataset
from libraries.model.dataset import dataloaderSingleSet
from libraries.model.options import Options
from libraries.utils import timeSpent
from libraries.model import postprocessing as pp

from time import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import unittest
#%%
def plotImages(images, figsize=(7,15), title=''):
    i=0
    for image in images:
        plt.figure(figsize=figsize)
        plt.title(title + ' n° {}'.format(i))
        plt.imshow(image)
        plt.show()
        i += 1
#%%
dataset = dataloaderSingleSet(1000, 1003, 1)
i = 1
#
plotImages(dataset.dataset.data)
plotImages(dataset.dataset.targets)
#%%
ckp = '/media/daniele/Data/Tesi/Thesis/Results/v1/Ganom_v1_v3_training_result/Ganom_v1_v3_best_ckp.pth.tar'
        
model = torch.load(ckp)
#%%
image = dataset.dataset.data[0]  
plt.imshow(image)
plt.show()

mask = dataset.dataset.targets[0]
plt.imshow(mask)
plt.show()

x, y = 500, 128
stride, patch_size = 10, 128
valid_patch, patch = nextPatch(image, x, y, stride, patch_size)
valid_patch, patch_mask = nextPatch(mask, x, y, stride, patch_size)
print('x: {} --> {}'.format(x, patch[1]))
print('y: {} --> {}'.format(y, patch[2]))
plt.imshow(patch[0])
plt.show()
plt.imshow(patch_mask[0])
plt.show()

#%%
start = time()
anom_map, mask_map = anomalyScoreFromImage(model, image, mask, 8, 32)
end = time()

timeSpent(end-start)
#%% PLOTS
# ANOMALY SCORES
plt.title('Anomaly Scores')
plt.imshow(anom_map)
plt.show()
# MASK
plt.title('Mask')
plt.imshow(mask_map)
plt.show()

#%% ANOMALY SCORE FROM DATASET
#dataset = dataloaderSingleSet(1000, 1005, 1)
as_map, gt_map = anomalyScoreFromDataset(model, dataset, 16, 32)
#%%
plotImages(as_map, title='Anomaly Score')
plotImages(gt_map, title='Mask')

#%% POSTPROCESSING
kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
conv = pp.convFilterScores(anom_map, kernel)

footprint = 4
med = pp.medFilterScores(anom_map, footprint)

plt.imshow(conv)
plt.show()

plt.imshow(med)
plt.show()

#%% POSTPROCESSING FROM DATASET

# CONVOLUTION
kernel = pp.createKernel(3, dim=3)
kernel.shape
convMap = pp.convFilterScores(as_map, kernel)
convMap.shape
plotImages(convMap, title='Conv Scores')

# MEDIAN
kernel_size = 3
medMap = pp.medFilterScores(as_map, kernel_size)
medMap.shape
plotImages(medMap, title='Med Scores')

# GAUSSIAN KERNEL
sigma = 3
gaussMap = pp.gaussFilterScores(as_map, sigma)
gaussMap.shape
plotImages(gaussMap, title='Gauss Scores')

#%%
class test_score(unittest.TestCase):
    
    def __init__(self):
        opt = Options()
        dataset = dataloaderSingleSet(opt, 1000, 1001)
        image = dataset.dataset.data[0]
        
        patch = nextPatch(image, 600, 200, 10, 64)
        
    def test_anomalyFromImage(self):
        opt = Options()
        dataset = dataloaderSingleSet(opt, 1000, 1001)
        image = dataset.dataset.data[0]
        
        a = anomalyScoreFromImage(5, image, 5, 20, 128)
        
unittest.main()     
#%%