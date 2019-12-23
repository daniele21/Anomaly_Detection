#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:55:40 2019

@author: daniele
"""
#%%
import torch
from torchvision import models
from libraries.torchsummary import summary
#%%
vgg = models.vgg16(pretrained=True).cuda()
vgg

#%%
summary(vgg, (3,256,1600))

#%%
modules = list(vgg.children())[:-1]

vgg.
