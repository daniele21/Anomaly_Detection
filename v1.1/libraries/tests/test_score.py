# -*- coding: utf-8 -*-

#%%
from libraries.model.score import nextPatch, anomalyScoreFromImage
from libraries.model.dataset import dataloaderSingleSet
from libraries.model.options import Options
from libraries.utils import timeSpent

from time import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import unittest
#%%
opt = Options()
dataset = dataloaderSingleSet(opt, 1000, 1001)
image = dataset.dataset.data[0]  
plt.imshow(image)
plt.show()

mask = dataset.dataset.targets[0]
plt.imshow(mask)
plt.show()

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
patch = nextPatch(image, x, y, stride, patch_size)
patch_mask = nextPatch(mask, x, y, stride, patch_size)
print('x: {} --> {}'.format(x, patch[1]))
print('y: {} --> {}'.format(y, patch[2]))
plt.imshow(patch[0])
plt.show()
plt.imshow(patch_mask[0])
plt.show()
#%%

patch = nextPatch(image, patch[1], patch[2], 1, 128)
print('x: {} --> {}'.format(patch[1], x))
print('y: {} --> {}'.format(patch[2], y ))
plt.imshow(patch[0])
plt.show()

#%%
start = time()
anom_map, mask_map = anomalyScoreFromImage(model, image, mask, 4, 32)
end = time()

timeSpent(end-start)
#%%
plt.imshow(anom_map)
plt.show()

plt.imshow(mask_map)
plt.show()
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
        