# -*- coding: utf-8 -*-

#%% IMPORTS
from network import Encoder, Decoder, Generator, Discriminator
from model_options import Options
from dataset import generateDataloader
from dataset import getCifar10 as cifar
#%% VARIABLE

opt = Options(in_channels=3)
opt.out_channels = 128

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