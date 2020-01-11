# -*- coding: utf-8 -*-

import torch
from torch.optim import Adam

from libraries.model.options import Options
from libraries.model.classifierFCONV import ClassifierFCONV
from libraries.model.classifierFCONN import ClassifierFCONNModel, ClassifierFCONN
from libraries.torchsummary import summary

from libraries.model.dataset import generateDataloaderFromDatasets
import libraries.model.dataset
from libraries.utils import Paths, Checkpoint, getAnomIndexes, computeAnomError, computeNormError
from libraries.utils import saveInfoAE
from libraries.model.postprocessing import distScores
from libraries.tests.test_result import automaticEvaluation
paths = Paths()

#%%
opt = Options()
model = ClassifierFCONV(opt)

summary(model.cuda(), (3,32,32))

#%%
model = ClassifierFCONN(opt)

summary(model.cuda(), (3,32,32))

#%%
opt = Options()
opt.name = 'FCONN_v0.0'
opt.anom_perc = 0.4
opt.in_channels = 3
opt.nFolders = 2
opt.patch_per_im = 500

opt.descr = '-----'
opt.descr = 'xavier init'
#%% DATASET

opt.augmentation = True
train, valid, test = dataset._setupDataset(opt, train='mixed', valid='mixed', test='mixed')

my_dataloader = generateDataloaderFromDatasets(opt, train, valid, test)

#%%
dataloader = my_dataloader
opt.dataset = 'steel dataset'

#dataloader = cifar_dataloader
#opt.dataset = 'cifar dataset'
#%%
trainloader = dataloader['train']
validLoader = dataloader['validation']
testloader = dataloader['test']
    
#%%
optimizer = Adam
model = ClassifierFCONNModel(opt, optimizer, trainloader, validLoader, testloader)
#aeModel.model.apply(weights_init)

opt.epochs = 15
opt.patience = 5
#opt.lr = 6.3*1e-5
opt.lr = 1e-05
model.train_model(opt)


#%%
model = aeModel
distScores(model.anomaly_scores, model.gt_labels, model.performance, bins=100)


















