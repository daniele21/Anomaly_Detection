#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:14:00 2019

@author: daniele
"""
#%%

import torch
import pickle

from libraries.model.models import BinClassifierModel
from libraries.model.FCN import CNN, FCN
from libraries.torchsummary import summary
import libraries.model.dataset as dataset
from libraries.model.options import Options
from libraries.utils import Paths, ensure_folder

paths = Paths()
#%%
opt = Options()
opt.patch_per_im = 500
opt.nFolders = 60

train_set, valid_set, test_set = dataset._setupDataset(opt, train='mixed', valid='mixed', test='mixed')
dataloader = dataset.generateDataloaderFromDatasets(opt, train_set, valid_set, test_set)

#%% SAVE DATALOADER

ensure_folder(paths.dataloaders + '/FCN/')

filename = 'FCN_60-500-30k.pickle'

with open(paths.dataloaders + '/FCN/' + filename, 'wb') as f:
    pickle.dump(dataloader, f)
    
#%% LOAD DATALOADER
filename = 'FCN_60-500-30k.pickle'

with open(paths.dataloaders + '/FCN/' + filename, 'rb') as f:
    dataloader = pickle.load(f)
    
#%% FCN SUMMARY
net = FCN()
summary(net.cuda(), (3,32,32))

#%% CNN SUMMARY

cnn = CNN()
summary(cnn.cuda(), (3,32,32))

#%% CNN TRAINING 30 EPOCHS

cnn_model = CNN()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-03)

model = BinClassifierModel(cnn_model, optimizer, dataloader)

#model.train_model(30)        
model.train_model(10)  
      
model.inference(dataloader['test'], 1)
#%% FCN TRAINING 30 EPOCHS

fcn_model = FCN()
optimizer = torch.optim.Adam(fcn_model.parameters(), lr=1e-03)

model = BinClassifierModel(fcn_model, optimizer, dataloader)

#model.train_model(30)        
model.train_model(10)  
      

#%%
x = torch.ones([64,3,32,32])
x.shape
model = CNN()
out = model(x)
#%% CONVERT A CNN TO FCN

fcn = FCN().cuda()

cnn = model.model
fcn.conv1.weight = cnn.conv1.weight
fcn.conv1.bias = cnn.conv1.bias
fcn.conv2.weight = cnn.conv2.weight
fcn.conv2.bias = cnn.conv2.bias
fcn.conv_fin.weight = cnn.fc2.weight.reshape((16, 256, 6,6))
fcn.fully_conv.weight = torch.reshape(cnn.fc3.weight, (1, 256, 1,1))













#%%
opt = Options()
opt.patch_per_im = 500
opt.nFolders = 60

train_set, valid_set, test_set = dataset._setupDataset(opt, train='mixed', valid='mixed', test='mixed')
dataloader = dataset.generateDataloaderFromDatasets(opt, train_set, valid_set, test_set)