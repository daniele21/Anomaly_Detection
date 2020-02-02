# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np

from libraries.model.network import FilterNN
from libraries.model.loss import binaryCrossEntropy_loss
from libraries.utils import plotMetrics
#%%

class FilterModel():
    
    def __init__(self, optmizer, trainloader, validloader):
        
        self.model = FilterNN()   
        self.optim = optmizer
        self.loss_function = binaryCrossEntropy_loss()
        self.trainloader = trainloader
        self.validloader = validloader

    def train_one_epoch(self):
        
        self.model.train()
        
        losses = []
        correct = 0
        total = 0
        
        for images, labels in self.trainloader:
            
            x = torch.Tensor(images).cuda()
            labels = torch.Tensor(labels).cuda()
            
            # FORWARD
            
            out = self.model(x)
            out = out.reshape(-1)
            loss = self.loss_function(out, labels)
            
            # BACKWARD
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # Evaluation
            losses.append(loss.item() * x.size(0))
        
            predictions = np.round(out.detach().cpu())
            targets = np.round(labels.detach().cpu())
            
            correct += (predictions == targets).sum().item()
            total += labels.size(0)
        
        loss_value = np.mean(losses)
        accuracy_value = correct/total
        
        return {'LOSS':loss_value, 'ACC':accuracy_value}
    
    def validation(self):
        
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
                
            for images, labels in self.validloader:
                
                x = torch.Tensor(images).cuda()
                labels = torch.Tensor(labels).cuda()
                
                out = self.model(x)
                out = out.reshape(-1)
                loss = self.loss_function(out, labels)
                
                predictions = np.round(out.cpu())
                targets = np.round(labels.cpu())
                
                correct += (predictions == targets).sum().item()
                total += labels.size(0)
            
                losses.append(loss.item() * x.size(0))
                
            loss_value = np.mean(losses)
            accuracy_value = correct/total
        
        return {'LOSS':loss_value, 'ACC':accuracy_value}
        
        
    def training_step(self):
        
        metrics = self._train_one_epoch()
        self.train['LOSS'].append(metrics['LOSS'])
        self.train['ACC'].append(metrics['ACC'])
        print('|>- Training:\tLoss: {:.3f} \t Accuracy: {:.3f}'.format(metrics['LOSS'],
                                                                   metrics['ACC']))

        metrics = self._validation()
        self.valid['LOSS'].append(metrics['LOSS'])
        self.valid['ACC'].append(metrics['ACC'])
        print('|>- Validation:\tLoss: {:.3f} \t Accuracy: {:.3f}'.format(metrics['LOSS'],
                                                                     metrics['ACC']))
        print('|')
        
    def train_model(self, epochs):
        
        self.train = {'LOSS':[], 'ACC':[]}
        self.valid = {'LOSS':[], 'ACC':[]}
        
        plot_rate = 5
        
        for epoch in range(epochs):
            print('\n************************************')
            print('> Epoch {}/{}'.format(epoch, epochs-1))
            print('|')
            
            self.training_step()
            
            # PLOTTING METRICS
            if(epoch % plot_rate == 0 and epoch!=0):
                plotMetrics()
                
    def kernel(self):
        
        return self.model.conv.weight
        
        
        
        
        
        
        
        