#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:55:40 2019

@author: daniele
"""
#%%
import torch
import torch.nn as nn
from torchvision import models
from libraries.torchsummary import summary
from libraries.model.options import Options, FullImagesOptions
from libraries.model.network import FCN_Generator, EncoderTL, GeneratorTL
from libraries.model.dataset import dataloaderPatchMasks

from matplotlib import pyplot as plt
#%%
opt = Options()
encTL = GeneratorTL(opt)
encTL
#%%
summary(encTL.cuda(), (3, 32, 32))

#%%
vgg = models.vgg16(pretrained=True).cuda()
vgg
#%%
for param in vgg.parameters():
    param.require_grad = False
#%%
summary(vgg, (3,32,32))

#%%
modules = list(vgg.children())[:-2]
modules.append(nn.Sequential(nn.Conv2d(512,100,(8,50))))
modules
#%%
modules = list(vgg.children())[:-2][0]
features = list(modules)[:-3]
encoder = nn.Sequential(*features)
encoder.add_module('Final conv2D', nn.Conv2d(512, 1024, 2))
encoder

#%%
encoder = nn.Sequential(*features)
encoder
#%%
summary(encoder.cuda(), (3,32,32))

#%%

encoder = nn.Sequential(*modules)
encoder

#%%

summary(encoder.cuda(), (3,256,1600))


#%%
k1, k2 = 3,3

decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(512, 256, (k1,k2), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(256, 128, (k1,k2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(128, 64, (k1,k2), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(64, 32, (k1,k2), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(32, 16, (k1,k2), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            
#            nn.ConvTranspose2d(32, 3, (2,2), stride=2),
##            nn.Tanh(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
            
            )
#%%

summary(decoder.cuda(), (512,8,50))
k1, k2 = 4, 25

decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, (k1,k2)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(1024, 512, (k1,k1), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(512, 256, (k1,k1), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(256, 128, (k1,k1), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(128, 64, (k1,k1), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(64, 32, (k1,k1), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(32, 3, (2,2), stride=2),
#            nn.Tanh(),
#            nn.Conv2d(3, 1, (256,1600)),
            nn.Tanh()
            
            )

summary(decoder.cuda(), (100,1,1))
#%%
modules = list(decoder.children())[:-1]
modules
a = nn.Sequential(*modules)

#%%

encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 128, 4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 256, 4, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.Conv2d(256, 512, 4, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            
            nn.Conv2d(512, 100, (6,48)),
#            nn.BatchNorm2d(1024),
#            nn.LeakyReLU(),
            
#            nn.Conv2d(1024, 1024)
            
            )


summary(encoder.cuda(), (3,256,1600))
#%%
opt = Options()
dec = Decoder(opt)
dec

summary(dec.cuda(), (100,1,1))

#%%
model = FCN_Generator().cuda()
model.encoder1
model.encoder2
model.decoder
model.finalLayer
model.fullyConvLayer
#%%

fullOpt = FullImagesOptions(start=0, end=100, batch_size=16)
dataloader = dataloaderPatchMasks(fullOpt)

#%%
i = 0

image = dataloader['test'].dataset.data[i]
plt.imshow(image)
plt.show()

label = dataloader['test'].dataset.targets[i]
plt.imshow(label)
plt.show()

#%%
from torchvision import models

r18 = models.resnet18(pretrained=True)


