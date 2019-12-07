#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:20:04 2019

@author: daniele
"""
#%% IMPORTS

from libraries.model.dataset import generateDataloader, loadDataset, SteelDataset, _setupDataset, generateDataloaderFromDatasets
from libraries.model.dataset import collectAnomalySamples, collectNormalSamples
from libraries.model import dataset
from libraries.model.options import Options
from libraries.utils import Paths
paths = Paths()
from matplotlib import pyplot as plt
import torch
import numpy as np
import unittest
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