# -*- coding: utf-8 -*-

#%% IMPORTS
from libraries.model.network import Encoder, EncoderTL, Decoder, Generator, Discriminator
from libraries.model import network 
from libraries.model.options import Options
from torchvision import models
from libraries.torchsummary import summary
import torch.nn as nn
#from dataset import generateDataloader
#from dataset import getCifar10 as cifar
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

opt.tl = 'resnet18'
#opt.tl = 'vgg16'
encoderTL = EncoderTL(opt)
encoderTL
summary(encoderTL.cuda(), (3,224,224))
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
#%%
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(512, 100)
summary(resnet.cuda(), (3,224,224))


#%%
resnet = models.resnet18(pretrained=True)

modules = list(resnet.children())[:-2]
modules = modules[:-1]
modules
net = nn.Sequential(*modules)
net.add_module('Final_Conv', nn.Conv2d(256, 100, 3, 2, 1))

summary(net.cuda(), (3,32,32))

