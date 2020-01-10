#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:20:04 2019

@author: daniele
"""
#%% IMPORTS

from libraries.model.dataset import generateDataloader, loadDataset, SteelDataset, _setupDataset, generateDataloaderFromDatasets
from libraries.model.dataset import collectAnomalySamples, collectNormalSamples, dataloaderFullImages
from libraries.model import dataset
from libraries.model.options import Options, FullImagesOptions
from libraries.utils import Paths
paths = Paths()
from matplotlib import pyplot as plt
import torch
import numpy as np
import unittest
from matplotlib import pyplot as plt
#%%

class test_dataset(unittest.TestCase):

    def test_loadData_normal(self):
        opt = Options()
        opt.patch_per_im = 10
        opt.nFolders = 5        
        
        train, valid, test = loadDataset(opt, test='normal')
        
        for label in valid['LABELS']:
            self.assertFalse(label)

    def test_loadData_mixed(self):
        opt = Options()
        opt.patch_per_im = 10
        opt.nFolders = 5        
        
        train, valid, test = loadDataset(opt, test='mixed')
        
        self.assertGreater(len(np.where(valid['LABELS'])), 0)
        
        self.assertGreater(len(np.where(valid['LABELS'])), 0)
        
#    def test_loadData_grayscale(self):
#        opt = Options()
#        opt.patch_per_im = 10
#        opt.nFolders = 5        
#        opt.in_channels = 1        
#        
#        train, valid, test = loadDataset(opt, test='mixed')
#        
#        for image in train['DATA']:
#            self.assertEqual(len(image.shape), 2)
#        
#        for image in valid['DATA']:
#            self.assertEqual(len(image.shape), 2)
#        
#        for image in test['DATA']:
#            self.assertEqual(len(image.shape), 2)
            
    def test_getItem_dataset(self):
        opt = Options()
        opt.patch_per_im = 10
        opt.nFolders = 5     
        opt.in_channels=1
        
        opt.loadDatasets()
        
        train = SteelDataset(opt, train=True)
        
        image, _ = train.__getitem__(5)
        
        self.assertIsNotNone(image)
        self.assertEqual(image.shape[0], 1)
              
unittest.main()
#%%
class test_loaddata(unittest.TestCase):
    
    def test_testset(self):
        opt = Options()
        opt.patch_per_im = 500
        opt.nFolders = 5     
        opt.in_channels=3
        
        opt.loadDatasets()
    
        test = opt.test_set
    
        
    
unittest.main()
#%%
class testFullDataloader(unittest.TestCase):
    
    def test_splitPatches(self):
        opt = FullImagesOptions()
        opt.start = 0
        opt.end = 20
        
        dataloader = dataloaderFullImages(opt)
        
        for i in range(1):
            im = dataloader['test'].dataset.data[i]
            mask = dataloader['test'].dataset.targets[i]
            
            plt.imshow(im)
            plt.show()
            plt.imshow(mask)
            plt.show()
            
            patches, patch_masks = dataset._splitPatches(im, mask)
        
            for j in range(20):
                plt.imshow(patches[j])
                plt.show()
                plt.imshow(patch_masks[j])
                plt.show()
#%%
class test_dataloader_patchmask(unittest.TestCase):    
    
    def test_dataloader_patchmasks(self):
        opt = FullImagesOptions()
        opt.start = 0
        opt.end = 2
        
        dataloader = dataset.dataloaderPatchMasks(opt)
        
        for i in range(5):
            patch = dataloader['train'].dataset.data[i]
            mask = dataloader['train'].dataset.targets[i]
            
            plt.imshow(patch)
            plt.show()
            plt.imshow(mask)
            plt.show()
            
    def test_masks(self):
        opt = FullImagesOptions()
        opt.start = 0
        opt.end = 10
        
        dataloader = dataset.dataloaderPatchMasks(opt)

        for _, labels in dataloader['train']:
            max_value = torch.max(labels)
            min_value = torch.min(labels)
            
            self.assertLessEqual(max_value, 1)
            self.assertGreaterEqual(min_value, 0)

unittest.main()
#%%
opt = FullImagesOptions()
dataloader = dataloaderFullImages(opt)

#%%
opt = Options()
opt.nFolders=5
opt.patch_per_im=500
opt.in_channels=3
opt.loadDatasets()

test = opt.test_set
labels = test['LABELS']
len(np.where(labels)[0])

dataloader = generateDataloader(opt)
test_loader = dataloader['test']
labels_loader = test_loader.dataset.targets
len(np.where(labels_loader)[0])
#%%
train = SteelDataset(opt, train=True)

image = train.__getitem__(3)

#%%
opt = Options()
opt.patch_per_im = 100
opt.nFolders = 10
training, validation, test = _setupDataset(opt, train='mixed', valid='mixed', test='mixed')
#training, validation, test = _setupDataset(opt, train='normal', valid='normal', test='mixed')

a = generateDataloaderFromDatasets(opt, training, validation, test)
#%%

a = dataset.createAnomalousPatches(100)