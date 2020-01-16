# -*- coding: utf-8 -*-

#%% IMPORTS
from libraries.model.network import Encoder, Decoder, Generator, Discriminator
from libraries.model.options import Options
from libraries.model.dataset import generateDataloader
from libraries.model.dataset import getCifar10 as cifar
from libraries.torchsummary import summary
#%% VARIABLE

opt = Options(in_channels=3)
opt.out_channels = 64

#%%
dataloader = cifar()

#%%
for elem in dataloader['train']:
    print(elem[0].shape)
    item = elem[0]
    break

#%% TEST ENCODER

encoder = Encoder(opt)    
encoder
summary(encoder.cuda(), (3,32,32))

#%% TEST DECODER

decoder = Decoder(opt)  

#%% TEST GENERATOR

Generator(opt)

#%%
discr = Discriminator(opt)
discr

classifier, features = discr(item)

print(len(classifier))
print(len(features))