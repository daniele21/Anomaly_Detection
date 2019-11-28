#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 19:35:19 2019

@author: daniele
"""



#%% IMPORTS
import torch
from torchvision import transforms as Transforms
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt

from libraries.model.options import Options
from libraries.model.generator import GeneratorModel
from libraries.model.network import weights_init
from libraries.model.dataset import generateDataloader, getCifar10
from libraries.utils import Paths, Checkpoint

paths = Paths()
#%%

opt = Options()
opt.name = 'Generator_v1_shuffled'
opt.in_channels=3

#%% DATASET

cifar_dataloader = getCifar10(opt)
my_dataloader = generateDataloader(opt)

#%%
dataloader = my_dataloader
#dataloader = cifar_dataloader

trainloader = dataloader['train']

try:
    validLoader = dataloader['test']
except:    
    validLoader = dataloader['validation']
    
#%%
optimizer = Adam
genModel = GeneratorModel(opt, optimizer, trainloader, validLoader)
genModel.model.apply(weights_init)


#%% LOAD MODEL
nome_ckp = 'Autoencoder_my_data_v1_lr:0.001|Epoch:29|Loss:0.036.pth.tar'
path_file = paths.checkpoint_folder + nome_ckp
ckp = torch.load(path_file)

genModel.loadModel(ckp)


#%% TRAINING MODEL
opt.epochs = 20
opt.patience = 5
#opt.lr = 6.3*1e-5
opt.lr = 0.0005
genModel.train_autoencoder(opt)

#%% PREDICTION

# NORMAL SAMPLES
i = 0
#%%
image = trainloader.dataset.data[i]
image.shape
genModel.predict(image)
i += 1
#%%
# ANOMALY SAMPLES
i = 1000
#%%
image = validLoader.dataset.data[i]
genModel.predict(image)
i += 1
#%% PLOTTING
genModel.avg_loss['validation'][-1]
genModel.plotting()

#%% TUNING

genModel.tuneLearningRate()

#%% SAVING CHECKPOINT

a = Checkpoint('model', 'optim', 12.23, 5, 'file.htiwod', '---')
a.save()

#%% LOADING EXISTENT MODEL

path_file = paths.checkpoint_folder + 'My Ganomaly_ckp_19_6.849.pth.tar'
path_file
a = torch.load(path_file)
a

model = a.model
optimizer = a.optimizer
type(optimizer)

#%% PREDICTION
image = validLoader.dataset.data[10]
image.shape
label = validLoader.dataset.targets[10]
label.shape
label

image_tensor = torch.FloatTensor(image)

image_tensor.shape
image_tensor = image_tensor.unsqueeze_(0)
image_tensor.shape
image_tensor

image_var = Variable(image_tensor).cuda()
image_var.shape

plt.imshow(image)
#%%
with torch.no_grad():
        pred = model(image_var)

pred.shape

output = pred.cpu().numpy()
output[0].shape
output = np.transpose(output[0], (2,1,0))
#output = output.view(0, 2,1,1)
output.shape

#output = 0.5 * (output + 1)
final_output = (output * 0.5) / 0.5
final_output = np.flip(final_output, 1)
final_output = np.rot90(final_output, 1)

output.shape
#plt.imshow(final_output)

fig, [ax1, ax2] = plt.subplots(2,1, figsize=(5,5))
#plt.figure()
ax1.imshow(image)
ax2.imshow(final_output)
plt.show()

#%%
for data in trainloader:
    
    break

#%%

# IMAGES
data[0]
data[0].shape

# LABELS
data[1]
data[1].shape

tensor_image = torch.Tensor(data[0]).cuda()
tensor_image.shape
tensor_image
