#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:27:19 2019

@author: daniele
"""

#%% IMPORTS
import torch
import torch.nn as nn
#import torchvision

import libraries.model.dataset as dataset
from libraries.model.options import Options
#from libraries.model.network import network
from libraries.torchsummary import summary

#%%
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        in_channels = 3
        out_channels = 8
        kernel_size = 3
        
        # LAYER 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        
        in_channels = out_channels
        out_channels *= 2
        
        # LAYER 2
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        
        in_channels = out_channels
        
        # FULLY CONNETTED
        self.fc1 = nn.Flatten()
        self.reluFC1 = nn.ReLU()
        self.fc2 = nn.Linear(14400, 256)
        self.reluFC2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)
#        self.fully_conv = nn.Conv2d(1024, 2, kernel_size=1)
        
    def forward(self, x):
        
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.pool1(h)
        
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool2(h)
        
        h = self.reluFC1(self.fc1(h))
        h = self.reluFC2(self.fc2(h))
        h = self.fc3(h)
        out = self.softmax(h)
#        h = self.fully_conv(h)
        
        return out



class FCN(nn.Module):
    
    def __init__(self):
        super(FCN, self).__init__()
        in_channels = 3
        out_channels = 8
        kernel_size = 3
        
        # LAYER 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        
        in_channels = out_channels
        out_channels *= 2
        
        # LAYER 2
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        
        in_channels = out_channels
        
        # FULLY CONVOLUTION
        self.conv_fin = nn.Conv2d(in_channels, 1024, kernel_size=30)
        self.fully_conv = nn.Conv2d(1024, 2, kernel_size=1)
        
    def forward(self, x):
        
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.pool1(h)
        
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool2(h)
        
        h = self.conv_fin(h)
        h = self.fully_conv(h)
        
        return h
    
#%%
net = FCN()
summary(net.cuda(), (3,128,128))

cnn = CNN()
summary(cnn.cuda(), (3,128,128))
#%%

dataloader

