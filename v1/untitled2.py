#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:56:55 2019

@author: daniele
"""
#%% IMPORTS

from libraries.model import network
import torch
import torch.nn as nn
from libraries.model.options import Options
#%%
opt = Options()

enc = network.Encoder(opt)
#%%
params = list(enc.parameters())
print(enc)
len(params)
#%%
for param in enc.parameters():
    print(param.shape)

#%%
gen = network.Generator(opt)

print(gen)
for param in gen.parameters():
    print(param.shape)
    
#%%
dec = gen.decoder

params = list(dec.parameters())
print(dec)
for param in dec.parameters():
    print(param.shape)
    
sel = params[-1]
sel.shape

#%%
layer = dec.net[-2:]
layer

#%%
for param in list(layer.parameters()):
    print(param.shape)
