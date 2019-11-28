# -*- coding: utf-8 -*-

#%%
from libraries.model.network import Discriminator
from libraries.model.options import Options
from libraries.model.dataset import generateDataloader, getCifar10

from torch import Tensor

#%%
opt = Options()

#%% DATASET

cifar_dataloader = getCifar10(opt)
#my_dataloader = generateDataloader(opt)

#%%

#dataloader = my_dataloader
dataloader = cifar_dataloader

trainloader = dataloader['train']

try:
    validLoader = dataloader['test']
except:    
    validLoader = dataloader['validation']
#%% MODEL

model = Discriminator(opt)
model

#%%    
image = trainloader.dataset.data[0]
image.shape
image_tensor = Tensor(image)

images = trainloader.dataset.data
images.shape
images_tensor = Tensor(images)
images.shape

for data in trainloader:
    
    break

batch = data[0]
batch.shape

classifier, feature = model(batch)
classifier.shape
feature.shape


