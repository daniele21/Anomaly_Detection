#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:14:00 2019

@author: daniele
"""
#%%

import torch
import pickle

from libraries.model.models import BinClassifierModel, FCNmodel
from libraries.model.FCN import CNN, FC_CNN, FCN
from libraries.torchsummary import summary
import libraries.model.dataset as dataset
from libraries.model.options import Options, FullImagesOptions
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

#%% FC_CNN SUMMARY
net = FC_CNN()
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

fc_cnn_model = FC_CNN()
optimizer = torch.optim.Adam(fc_cnn_model.parameters(), lr=1e-03)

model = BinClassifierModel(fc_cnn_model, optimizer, dataloader)

#model.train_model(30)        
model.train_model(10)  
      

#%% TESTING FCN

net = FCN()
summary(net.cuda(), (3,256,1024))

opt = FullImagesOptions()
opt.start = 0
opt.end = 256
opt.batch_size = 1
dataloader = dataset.dataloaderFullImages(opt)
    
fcn = FCN()
optimizer = torch.optim.Adam(fcn.parameters(), lr=1e-03)

fcn_model = FCNmodel(fcn, optimizer, dataloader)

fcn_model.train_model(10)


#%%
#labels = dataloader['test'].dataset.targets

torch.Tensor(labels.float())
#%%
for images, labels in dataloader['train']:
    break
#%%
    
labels.float()