# -*- coding: utf-8 -*-
#%%
from libraries.MultiTaskLoss import MultiLossWrapper
from libraries.model.options import Options
from libraries.model.dataset import generateDataloader, getCifar10
from libraries.model.dataset import collectAnomalySamples, collectNormalSamples
from libraries.model.adModel import AnomalyDetectionModel
from libraries.utils import Paths, getAnomIndexes, computeAnomError, computeNormError
paths = Paths()

from matplotlib import pyplot as plt
import numpy as np
import pickle

import torch
from torch.optim import Adam
from torchvision import transforms as Transforms
from torch.autograd import Variable
import sys
#%% LOAD OPTIONS
opt = Options()
opt.name = 'Ganom_1_prova'

#opt.in_channels = 1 # GRAYSCALE
opt.in_channels = 3 # RGB

opt.nFolders = 60
opt.patch_per_im = 500

#opt.nFolders = 30
#opt.patch_per_im = 1000

#opt.nFolders = 60
#opt.patch_per_im = 1500

#opt.n_extra_layers=2

opt.batch_size = 64
#opt.out_channels = 128

opt.descr = '-----'
opt.descr = 'augmentation - validation con norm - no weighting losses - thr over norm/anom'

#%% GENERATE DATASET

opt.augmentation = True
my_dataloader = generateDataloader(opt)

#%% LOAD DATASET

with open(paths.dataloaders + 'v1_aug_60-500-30k.pickle', 'rb') as data:
    my_dataloader = pickle.load(data)

#%% SAVE DATASET
filename = 'v1_60-500-30k.pickle'
with open(paths.dataloaders + '/v1' + filename, 'wb') as f:
    pickle.dump(my_dataloader, f)
#%%

dataloader = my_dataloader
opt.dataset = 'steel dataset'

#%%
trainloader = dataloader['train']
validLoader = dataloader['validation']
testloader = dataloader['test']
    
#%% MODEL OPTIONS
optimizer_gen = Adam
optimizer_discr = Adam

opt.lr_gen = 1*1e-04
opt.lr_discr = 1*1e-04

opt.patience = 5

opt.w_adv = 1
opt.w_con = 50
opt.w_enc = 1

opt.weightedLosses = False

epochs = 30

#%% TUNING MODEL
tuning = adModel.tuneLearningRate(-6, -7, -6, -7)

#%% LOAD MODEL
optimizer_gen = Adam
optimizer_discr = Adam

adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr) 

opt.name = 'Ganom_v12.0'
nome_ckp = 'Ganom_v12.0_lr:1e-06|Epoch:201|Auc:0.901|Loss:178.0066.pth.tar'
path_file = paths.checkpoint_folder + opt.name + '/' + nome_ckp
print(path_file)
#adModel.loadCheckPoint(path_file)
adModel = torch.load(path_file)
#%% MULTI TASK LOSS WEIGHTS
adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr, optimizer_gen,
                                trainloader, validLoader, testloader) 

mtl = MultiLossWrapper(adModel, trainloader, 3)
optim = torch.optim.Adam(mtl.multiTaskLoss.parameters(), lr=1e-03)
mtl.train(10, optim)
#%% TRAINING MODEL

adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr, optimizer_gen,
                                trainloader, validLoader, testloader) 

adModel.train_model(1)


#%%
opt.lr_gen = 1*1e-04
#opt.lr_discr = 5*1e-05

adModel.opt.patience = 5
new_lr = 1e-04
adModel.model.optimizer_gen.param_groups[0]['lr'] = new_lr

adModel.resumeTraining(100)
#%% RESULTS
adModel.plotting()
adModel.evaluateRoc()

#%% TEST SET ERROR
anom_set = collectAnomalySamples(100)
norm_set = collectNormalSamples(90, 80)
#%% PREDICT

i = 0
#%%
image = validLoader.dataset.data[i]
target = validLoader.dataset.targets[i]
adModel.predict(image, target, i, verbose=1)
adModel.predict(image, target, 'nonono', verbose=1)
#print(score)
i += 1
#%% ANOMALY
anom_index = getAnomIndexes(validLoader.dataset.targets)
i = 0
#%%
image = validLoader.dataset.data[anom_index[i]]
target = validLoader.dataset.targets[anom_index[i]]
adModel.predict(image, target, i, verbose=1)
i += 1

#%% ANOM ERROR
anomalyError = computeAnomError(adModel, anom_set)
normalError = computeNormError(adModel, norm_set)

content = '- Anom_Error: {:.3f}'.format(anomalyError)
adModel.addInfo(content)

content = '\n- Norm_Error: {:.3f}'.format(normalError)
adModel.addInfo(content)

#%%
