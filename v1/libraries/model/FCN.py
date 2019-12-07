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
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        
        in_channels = out_channels
        out_channels *= 2
        
        # LAYER 2
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        
        in_channels = out_channels
        
        # FULLY CONNETTED
        self.fc1 = nn.Flatten()
        self.reluFC1 = nn.ReLU()
        self.fc2 = nn.Linear(576, 256)
        self.reluFC2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)
        self.softmax = nn.Sigmoid()
#        self.fully_conv = nn.Conv2d(1024, 2, kernel_size=1)
        
    def forward(self, x):
        
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.pool1(h)
        
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool2(h)
#        
        h = self.reluFC1(self.fc1(h))
        h = self.reluFC2(self.fc2(h))
        h = self.fc3(h)
        out = self.softmax(h)
        
        return out

class FCN(nn.Module):
    
    def __init__(self):
        super(FCN, self).__init__()
        in_channels = 3
        out_channels = 32
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
    
class CNNmodel():
    
    def __init__(self, dataloader):
        
        self.model = CNN().cuda()
        self.trainloader = dataloader['train']
        self.validloader = dataloader['validation']
        self.testloader = dataloader['test']
        
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-03)
        
    def _train_one_epoch(self):
        
        losses = []
        correct = 0
        total = 0
        
        for images, labels in self.trainloader:
            
            x = torch.Tensor(images).cuda()
#            print(x.shape)
            labels = torch.Tensor(labels).cuda()
#            print(labels.shape)
#            print(labels)
            
            output = self.model(x)
#            print(output.shape)
#            print(output)
            
            loss = self.loss_function(output, labels)
#            print(loss.item())
#            print(x.size(0))
            correct += (output == labels).sum().item()
            total += labels.size(0)
#            print((output == labels).sum().item())
#            print(labels.size(0))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            losses.append(loss.item() * x.size(0))
#            print(loss.item() * x.size(0))
            
        loss_value = np.mean(losses)
        accuracy_value = 100*(correct/total)
        
        return loss_value, accuracy_value
    
    def _validation(self):
        
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
                
            for images, labels in self.validloader:
                
                x = torch.Tensor(images).cuda()
                labels = torch.Tensor(labels).cuda()
                
                output = self.model(x)
                loss = self.loss_function(output, labels)
                correct += (output == labels).sum().item()
                
                total += labels.size(0)
            
                losses.append(loss.item() * x.size(0))
                
                
            loss_value = np.mean(losses)
            accuracy_value = 100*(correct/total)
        
        return loss_value, accuracy_value
        
    def train_model(self, epochs):
        
        train_losses = []
        train_accs = []
        
        valid_losses = []
        valid_accs = []
        
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs-1))
            train_loss, train_acc = self._train_one_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            valid_loss, valid_acc = self._validation()
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            
            plt.title('Loss')
            plt.plot(train_losses, c='r', label='Training')
            plt.plot(valid_losses, c='b', label='Validation')
            plt.legend()
            plt.show()
            
            plt.title('Accuracy')
            plt.plot(train_accs, c='r', label='Training')
            plt.plot(valid_accs, c='b', label='Validation')
            plt.legend()
            plt.show()

#%%
opt = Options()
opt.patch_per_im = 500
opt.nFolders = 60

train_set, valid_set, test_set = dataset._setupDataset(opt, train='mixed', valid='mixed', test='mixed')
dataloader = dataset.generateDataloaderFromDatasets(opt, train_set, valid_set, test_set)
#%%
model = CNNmodel(dataloader)   
model.train_model(30)        

#%%
net = FCN()
summary(net.cuda(), (3,32,32))
#%%
cnn = CNN()
summary(cnn.cuda(), (3,32,32))
#%%
opt = Options()
opt.patch_per_im = 500
opt.nFolders = 60

train_set, valid_set, test_set = dataset._setupDataset(opt, train='mixed', valid='mixed', test='mixed')
dataloader = dataset.generateDataloaderFromDatasets(opt, train_set, valid_set, test_set)
