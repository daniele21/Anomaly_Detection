# -*- coding: utf-8 -*-
#%%
import torch
import torch.nn as nn

#%%
def _l1_loss(input, target):
    return torch.mean(torch.abs(input - target))
    
def _l2_loss(input, target, size_average=True):
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)
    
def adversial_loss(input, target):
    return _l2_loss(input, target)
        
#def encoder_loss(input, target):
#    return _l2_loss(input, target)

def encoder_loss(input, target):
    return _l2_loss(input, target)

def contextual_loss():
    return nn.L1Loss()

def binaryCrossEntropy_loss():
    return nn.BCELoss()

#%% TEST
    
#a = torch.Tensor([10])
#b = torch.Tensor([15])
##
#loss_function = contextual_loss()
#loss_function(a,b).item()
#
#loss_function = encoder_loss()
#loss_function(a,b).item()

