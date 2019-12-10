#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:27:19 2019

@author: daniele
"""

#%% IMPORTS
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchvision import models

#%%
class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        in_channels = 3
        out_channels = 64
        
        # LAYER 1        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # LAYER 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        
        in_channels = out_channels
        out_channels *= 2
        
        # LAYER 3
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)

        in_channels = out_channels
        out_channels *= 2
        
        # LAYER 4
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2,2)
        
        # FULLY CONNETTED 
        self.flattenNeurons = 1024
        
        self.fc1 = nn.Linear(self.flattenNeurons, 256)
        self.reluFC1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        h = self.conv1(x)
        h = self.relu1(h)
        
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool2(h)

        h = self.conv3(h)
        h = self.relu3(h)
        h = self.pool3(h)
        
        h = self.conv4(h)
        h = self.relu4(h)
        h = self.pool4(h)
        
        h = h.view(-1, self.flattenNeurons)
        h = self.reluFC1(self.fc1(h))
        out = self.sigmoid(self.fc2(h))
        
        return out

#%%
class FC_CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        in_channels = 3
        out_channels = 32
        
        # LAYER 1        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU()
        
        in_channels = out_channels
#        out_channels *= 2
        
        # LAYER 2
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU()
        
        in_channels = out_channels
#        out_channels *= 2
        
        # LAYER 3
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0)
        self.relu3 = nn.LeakyReLU()
        
        in_channels = out_channels
        out_channels *= 2
        
        # LAYER 4
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0)
        self.relu4 = nn.LeakyReLU()
        
        in_channels = out_channels
        out_channels *= 2
        
        # LAYER 5
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.relu5 = nn.ReLU()
        
        in_channels = out_channels
        
        # FINAL LAYER 
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        
        h = self.relu1(self.conv1(x))
        h = self.relu2(self.conv2(h))
        h = self.relu3(self.conv3(h))
        h = self.relu4(self.conv4(h))
        h = self.relu5(self.conv5(h))
        h = self.activation(self.final_conv(h))

        return h

class FCN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        in_channels = 3
        out_channels = 32
        kernel_size = 3
        
        # LAYER 1        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU()
        
        in_channels = out_channels
        #        out_channels *= 2
        
        # LAYER 2
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU()
        
        in_channels = out_channels
        #        out_channels *= 2
        
        # LAYER 3
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0)
        self.relu3 = nn.LeakyReLU()
        
        in_channels = out_channels
        out_channels *= 2
        
        # LAYER 4
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0)
        self.relu4 = nn.LeakyReLU()
        
        in_channels = out_channels
        out_channels *= 2
        
        # FC1
        self.fc1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.reluFC1 = nn.ReLU()
        
        in_channels = out_channels
        
        # FC2
        self.fc2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.reluFC2 = nn.ReLU()
        
        # SCORE
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        
        # UPSAMPLE 
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=64, stride=32)
        
    def forward(self, x):
        
        h = self.relu1(self.conv1(x))
        h = self.relu2(self.conv2(h))
        h = self.relu3(self.conv3(h))
        h = self.relu4(self.conv4(h))
        
        h = self.reluFC1(self.fc1(h))
        h = self.reluFC2(self.fc2(h))
        
        h = self.score(h)
        
        h = self.upsample(h)
#        print(h.shape)
#        print(x.size())
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
#        print(h.shape)
        return h
#%%
class pretrainedModel():
    
    def __init__(self, dataloader):
        
        self.pre_model = models.vgg19(pretrained=True).cuda()
        self.dataloader = dataloader
        self.freezeParams()
        self.setModel()
        
        
    def freezeParams(self):
        
        for param in self.pre_model.parameters():
            param.requires_grad = False
    
    def setModel(self):
        num_ftrs = self.pre_model.fc.in_features
        self.pre_model.fc = nn.Linear(num_ftrs, 2)
    
    def train(self):
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.pre_model.parameters(), lr=0.001, momentum=0.9)  


                      



