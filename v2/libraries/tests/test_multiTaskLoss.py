# -*- coding: utf-8 -*-

#%% IMPORTS
import torch

from libraries.model.options import Options
from libraries.model.adModel import AnomalyDetectionModel
from libraries.MultiTaskLoss import MultiLossWrapper
#%%
opt = Options()
optimizer = torch.optim.Adam

dataloader = my_dataloader

trainloader = dataloader['train']
validLoader = dataloader['validation']
#%%
adModel = AnomalyDetectionModel(opt, optimizer, optimizer, trainloader, validLoader)
#%%

wrapper = MultiLossWrapper(adModel, trainloader, 3)

wrapper.train(20, optimizer)
    
wrapper.plotLoss()
