#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:50:36 2019

@author: daniele
"""

#%%
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
import pylab
from time import time

from libraries.utils import EarlyStopping

device = torch.device('cuda:0')
#%%

class MultiTaskLoss(nn.Module):
    '''
    Description:
        
    Params:
        - model --> model where the loss are applied to
        - losses -> dict{'loss_name' : loss_value}
    
    '''
    
    def __init__(self, model, n_losses):
        super().__init__()
        self.model = model
        self.log_vars = nn.Parameter(torch.zeros( (n_losses) ) )
#        self.log_vars = nn.Parameter(torch.Tensor( [1., 1., 1.]) )
        
    def forward(self, x):
        
        x_prime, z, z_prime = self.model.forward_gen(x)
        _, feat_real, _, feat_fake = self.model.forward_discr(x, x_prime)
        
        _, [loss_adv, loss_con, loss_enc] = self.model.loss_function_gen(x, x_prime, z, z_prime, feat_fake, feat_real)
        
        factor_adv = torch.exp(-self.log_vars[0])
        weighted_loss_adv = factor_adv * loss_adv + self.log_vars[0]
        loss = torch.sum(weighted_loss_adv , -1)
        
        factor_con = torch.exp(-self.log_vars[1])
        weighted_loss_con = factor_con * loss_con + self.log_vars[1]
        loss += torch.sum(weighted_loss_con , -1)
        
        factor_enc = torch.exp(-self.log_vars[2])
        weighted_loss_enc = factor_enc * loss_enc + self.log_vars[2]
        loss += torch.sum(weighted_loss_enc, -1)
        
#        loss = torch.mean(loss)
        
        factors = {'ADV':factor_adv, 'CON':factor_con, 'ENC':factor_enc}
        
#        print('Log vars')
#        print(self.log_vars.data.tolist())
#        print('\n')
#        print('w_loss_adv: {}'.format(weighted_loss_adv))
#        print('w_loss_con: {}'.format(weighted_loss_con))
#        print('w_loss_enc: {}'.format(weighted_loss_enc))
        return loss, self.log_vars.data.tolist(), factors


class MultiLossWrapper():
    
    def __init__(self, model, dataloader, n_losses):
        self.multiTaskLoss = MultiTaskLoss(model, n_losses).to(device)
        self.data = dataloader
        self.optimizer = torch.optim.Adam(self.multiTaskLoss.parameters(), lr=1e-3)
        
    def train(self, epochs, patience=2, verbose=1):
        print('Multi loss weighting')
        self.multiTaskLoss.train()
        self.es = EarlyStopping(patience)
        self.loss = []
        
        start = time()
        self.activeParams()
        for epoch in range(epochs):
            print('\n\nMulti-loss Epoch {}/{}'.format(epoch, epochs))
            
            loss_list = []
            
            for images, labels in self.data:
                
                x = torch.Tensor(images).cuda()
                labels = torch.Tensor(labels).cuda()
        
                loss, log_vars, self.factors = self.multiTaskLoss(x)
                
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
                loss_list.append(loss.item() * x.size(0))
        
            loss_mean = np.mean(loss_list)
            self.loss.append(loss_mean)
            self.es(loss_mean)
            
            if(self.es.early_stop):
                break
            
            if(verbose):
                print('\n')
                print('loss_model: \t{:.2f}\n'.format(loss_mean))
                
                print('Loss Weights:')
                print('w_adv: \t{:.2f}'.format(self.factors['ADV']))
                print('w_con: \t{:.2f}'.format(self.factors['CON']))
                print('w_enc: \t{:.2f}'.format(self.factors['ENC']))
                
                print('\nLog Vars:')
                print(log_vars)
        
        end = time()
        
        print('\n')
        print('loss_model: \t{:.2f}'.format(loss_mean))
        
        print('w_adv: \t{:.2f}'.format(self.factors['ADV']))
        print('w_con: \t{:.2f}'.format(self.factors['CON']))
        print('w_enc: \t{:.2f}'.format(self.factors['ENC']))
        
        print('\nloss_wrapper: {:.2f}'.format(loss_mean))
        
        minutes = (end - start) / 60
        print('Spent time: {:.3f}'.format(minutes))
#        self.plotLoss()
        
        print('\n')
        print('Freezing multi taks loss params')
        self.freezeParams()
            
        
        return self.factors['ADV'], self.factors['CON'], self.factors['ENC']
        
    def freezeParams(self):
        print('>- Multi Task Loss: FREEZING PARAMS ')
        self.multiTaskLoss.log_vars.requires_grad=False
    
    def activeParams(self):
        print('>- Multi Task Loss: SET TRAINABLE PARAMS ')
        self.multiTaskLoss.log_vars.requires_grad=True
            
    def plotLoss(self):
        pylab.plot(self.loss)
        pylab.show()

    def get_LogVars(self):
        return self.multiTaskLoss.log_vars
    
#%% 
a = MultiLossWrapper(adModel.model, trainloader, 3)
    
a.freezeParams()
a.activeParams()

a.train(20)




    