# -*- coding: utf-8 -*-
import torch
from libraries.model.filterModel import FilterModel
from libraries.model.dataset import generateDataloaderPerDefect
from libraries.model.options import Options
from libraries.model import postprocessing as pp

from matplotlib import pyplot as plt
#%%

opt = Options()
opt.in_channels = 1
opt.out_channels = 1
opt.batch_size = 10
n_samples = 10
filter_data = generateDataloaderPerDefect(opt, n_samples)

#%%

optim = torch.optim.Adam
trainloader = filter_data[1]
validloader = filter_data[1]

#%%
k = 5
opt.lr = 1e-02
model = FilterModel(optim, trainloader, validloader, opt, k)

model.train_model(40)

#%%
kernel = model.model.conv.weight
kernel = kernel.cpu().detach().numpy()
kernel = kernel.reshape(k,k)

plt.imshow(kernel)
kernel

#%%
image = trainloader.dataset.data[0]
plt.imshow(image)
plt.show()

label = trainloader.dataset.targets[0]
plt.imshow(label)
plt.show()

pp.convFilterScores()
