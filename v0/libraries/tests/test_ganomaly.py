# -*- coding: utf-8 -*-
#%%
from matplotlib import pyplot as plt
import numpy as np
import pickle

from torch.optim import Adam
import torch
from torchvision import transforms as Transforms
from torch.autograd import Variable

from libraries.model.options import Options
from libraries.model.dataset import generateDataloader, getCifar10
from libraries.model.dataset import collectAnomalySamples, collectNormalSamples
from libraries.model.ganomaly import AnomalyDetectionModel
from libraries.model.evaluate import evaluate
from libraries.utils import Paths, getAnomIndexes, computeAnomError, computeNormError
from libraries.tests.test_result import automaticEvaluation

paths = Paths()
#%% LOAD OPTIONS
opt = Options(endFolder=100)
opt.name = 'Ganom_v0.0'
opt.anom_perc = 0.4
#opt.in_channels = 1 # GRAYSCALE
opt.in_channels = 3 # RGB
opt.nFolders = 30
opt.patch_per_im = 100

#opt.n_extra_layers=2

opt.batch_size = 64
#opt.out_channels = 128

opt.descr = '-----'
opt.descr = 'xavier init'

#%% DATASET

#cifar_dataloader = getCifar10(opt)
my_dataloader = generateDataloader(opt)

#%% LOAD DATALOADER

filename = '../../variables/v0/v0_60-500-30k.pickle'
with open(filename, 'rb') as f:
    my_dataloader = pickle.load(f)
    
#%%

dataloader = my_dataloader
opt.dataset = 'steel dataset'

#dataloader = cifar_dataloader
#opt.dataset = 'cifar dataset'
#%%
trainloader = dataloader['train']

try:
    validLoader = dataloader['test']
except:    
    validLoader = dataloader['validation']
    
#%% MODEL
optimizer_gen = Adam
optimizer_discr = Adam

opt.lr_gen = 1*1e-04
opt.lr_discr = 1*1e-04
adModel = AnomalyDetectionModel(opt,optimizer_gen, optimizer_discr, trainloader, validLoader) 

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

#%% TRAINING MODEL

#opt.lr = 0.0001
#opt.lr_gen = 1e-05
#opt.lr_discr = 1e-05

opt.epochs = 10
opt.patience = 3

opt.w_adv = 1
opt.w_con = 50
opt.w_enc = 1

adModel.train_model()

#%% RESUME LEARNING

opt.lr_gen = 1*1e-05
new_lr = opt.lr_gen
adModel.model.optimizer_gen.param_groups[0]['lr'] = new_lr
adModel.model.optimizer_discr.param_groups[0]['lr'] = new_lr
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

#%% INFERENCE

automaticEvaluation(adModel, 1070, 1080, 12)










