#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:14:00 2019

@author: daniele
"""
#%%
import pickle
from libraries.model.FCN import CNNmodel, FCN, CNN
from libraries.torchsummary import summary
#%%
opt = Options()
opt.patch_per_im = 500
opt.nFolders = 60

train_set, valid_set, test_set = dataset._setupDataset(opt, train='mixed', valid='mixed', test='mixed')
dataloader = dataset.generateDataloaderFromDatasets(opt, train_set, valid_set, test_set)

#%%
ensure_folder(paths.dataloaders + '/FCN')

filename = 'FCN_60-500-30k.pickle'

with open(paths.dataloaders + '/FCN' + filename, 'wb') as f:
    pickle.dump(dataloader, f)
#%%
model = CNNmodel(dataloader)   
model.train_model(30)        

#%%
net = FCN()
summary(net.cuda(), (3,32,32))
#%%
cnn = CNN()
summary(cnn.cuda(), (3,32,32))
#%%

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