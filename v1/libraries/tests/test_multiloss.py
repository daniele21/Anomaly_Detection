#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:44:36 2019

@author: daniele
"""

#%% IMPORTS 

import libraries.model.network as net
from libraries.model.options import Options
from libraries.model.dataset import generateDataloader
import libraries.model.loss as loss
from libraries.utils import EarlyStopping

import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda:0')
#%%
opt = Options()

opt.in_channels = 3 # RGB
opt.nFolders = 2
opt.patch_per_im = 500

my_dataloader = generateDataloader(opt)

dataloader = my_dataloader
opt.dataset = 'steel dataset'
trainloader = dataloader['train']
validLoader = dataloader['validation']
testloader = dataloader['test']

#%%
class multiLoss(nn.Module):
    
#    def __init__(self, model, n_losses):
    def __init__(self, model, initLog):
        super().__init__()
        self.model = model
        self.log_vars = nn.Parameter(torch.Tensor( initLog) )
#        self.log_vars = nn.Parameter(torch.zeros((n_losses)) )
        self.factor_con = torch.exp(-self.log_vars[0])
        self.factor_enc = torch.exp(-self.log_vars[1])
        
    def initLogs(self, logs):
        self.log_vars = nn.Parameter(torch.Tensor([logs[0].item(), logs[1].item()]))
    
    def forward(self, x):
        
        x_prime, z, z_prime = self.model(x)
        
        self.factor_con = torch.exp(-self.log_vars[0])
        w_loss = self.factor_con * loss.contextual_loss()(x, x_prime) + self.log_vars[0]
        
        self.factor_enc = torch.exp(-self.log_vars[1])
        w_loss += self.factor_enc * loss.encoder_loss(z, z_prime) + self.log_vars[1]
        
#        factor = torch.exp(-self.log_vars[0])
#        w_loss = factor * loss.contextual_loss()(x, x_prime) + self.log_vars[0]
        
#        loss_mtl = torch.sum(w_loss, -1)
        loss_mtl = torch.sum(w_loss).cuda()
        
        return loss_mtl, self.log_vars.data.tolist(), [self.factor_con, self.factor_enc]
    
    def getFactor(self):
        return self.factor_con, self.factor_enc

class Wrapper():
    
#    def __init__(self, model, dataloader, n_losses):
#        self.multiLoss = multiLoss(model, n_losses).to(device)
#        self.data = dataloader
#        self.optimizer = torch.optim.Adam(self.multiLoss.parameters(), lr=1e-3)
#   
    def __init__(self, model, dataloader, initLog):
        self.multiLoss = multiLoss(model, initLog).to(device)
        self.data = dataloader
        self.optimizer = torch.optim.Adam(self.multiLoss.parameters(), lr=0.1)
    
    def initLogs(self, logs):
        self.multiLoss.initLogs(logs)
    
    def getFactor(self):
        return self.multiLoss.getFactor()
    
    def getLogVars(self):
        return self.multiLoss.log_vars
        
    def train(self, epochs, patience=2, verbose=1):
        print('\n-->Multi loss weighting')
        self.multiLoss.train()
        self.es = EarlyStopping(patience)
        self.loss = []
        
        for epoch in range(epochs):
            print('\n\nMulti-loss Epoch {}/{}'.format(epoch, epochs))
            
            loss_list = []
            
            for images, labels in self.data:
                
                x = torch.Tensor(images).cuda()
                labels = torch.Tensor(labels).cuda()
        
                loss_mtl, log_vars, self.factors = self.multiLoss(x)
                
#                print(loss_mtl)
                self.optimizer.zero_grad()
                loss_mtl.backward(retain_graph=True)
                self.optimizer.step()
                
                loss_list.append(loss_mtl.item() * x.size(0))
        
            loss_mean = np.mean(loss_list)
            self.loss.append(loss_mean)
            self.es(loss_mean)
            
            if(self.es.early_stop):
                break
            
            if(verbose):
                print('\n')
                print('loss_model: \t{:.2f}\n'.format(loss_mean))
                
                print('Loss Weights:')
                print('w_con: \t{:.2f}'.format(self.factors[0]))
                print('w_adv: \t{:.2f}'.format(self.factors[1]))
                
                print('\nLog Vars:')
                print(log_vars)
        
        
        print('\n')
        print('loss_model: \t{:.2f}'.format(loss_mean))
        
        print('w_con: \t{:.2f}'.format(self.factors[0]))
        print('w_adv: \t{:.2f}'.format(self.factors[1]))
        
#%%
gen = net.Generator(opt).cuda()
optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)


Logs = torch.Tensor([0., 0.])
wrapper = Wrapper(gen, trainloader, Logs)

epochs = 10

print('Training')
for epoch in range(epochs):
    print('\n\n>- Epoch {}/{}\n'.format(epoch, epochs))
    i = 0
    
#    wrapper.initLogs(Logs)
    wrapper.train(5)
    factor_con, factor_enc = wrapper.getFactor()
    Logs = wrapper.getLogVars()
    print(Logs)
    for images, labels in trainloader:
        print('------------------------------')
        print('>- Batch n. {}\n\n'.format(i))
        
        x = torch.Tensor(images).cuda()
        labels = torch.Tensor(labels).cuda()
        
        x_prime, z, z_prime = gen(x)
        
        con_loss = loss.contextual_loss()(x, x_prime)
        enc_loss = loss.encoder_loss(z, z_prime)
        loss_gen = factor_con * con_loss + factor_enc * enc_loss
    
        print('Factor_con: {}'.format(factor_con))
        print('Factor_enc: {}'.format(factor_enc))
        print('Loss: {:.3f}'.format(con_loss))
    
        optimizer.zero_grad()
        loss_gen.backward(retain_graph=True)
        optimizer.step()
        
        i += 1    
        
    print('Fine epoca')
    
        
#%%
gen = net.Generator(opt).cuda()
optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)

wrapper = Wrapper(gen, trainloader, 2)
        
wrapper.train(5)
        
#%%
a = torch.Tensor([1.])
a.requires_grad=True
b = a*5
b.backward(retain_graph=True)        
a.grad        


b.backward()        
a.grad        
        
        
#%%
gen = net.Generator(opt).cuda()
a = multiLoss(gen, [math.log(1.03), math.log(1.01)])
    
#%%
a.factor_con
a.factor_enc

a.log_vars
torch.exp(a.log_vars)


0-math.log(1.03)        
