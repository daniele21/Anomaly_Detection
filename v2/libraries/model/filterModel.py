# -*- coding: utf-8 -*-

#%%
import torch
import numpy as np

from libraries.model.network import FilterNN
from libraries.model.loss import binaryCrossEntropy_loss
from libraries.utils import plotMetrics
#from libraries.model import score
from libraries.model import postprocessing as pp

#%%

class FilterModel():
    
    def __init__(self, optmizer, trainloader, validloader, opt, k):
        
        self.model = FilterNN(opt, k).cuda()
        self.optim = optmizer(self.model.parameters(), opt.lr)
        self.loss_function = binaryCrossEntropy_loss()
        self.trainloader = trainloader
        self.validloader = validloader

    def _train_one_epoch(self, thr):
        
        self.model.train()
        
        losses = []
        correct = 0
        total = 0
        
        for as_filter, labels in self.trainloader:
            
            x = torch.Tensor(as_filter).cuda()
            labels = labels.reshape(-1)
            labels = torch.Tensor(labels.float()).cuda()
            
            # FORWARD
            out = self.model(x)
#            print(out)
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
    
    def _validation(self, thr):
        
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
                
            for images, labels in self.validloader:
                
                x = torch.Tensor(images).cuda()
                labels = labels.reshape(-1)
                labels = torch.Tensor(labels.float()).cuda()
                
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
        
        
    def _training_step(self, thr):
        
        metrics = self._train_one_epoch(thr)
        self.train['LOSS'].append(metrics['LOSS'])
        self.train['ACC'].append(metrics['ACC'])
        print('|>- Training:\tLoss: {:.3f} \t Accuracy: {:.3f}'.format(metrics['LOSS'],
                                                                   metrics['ACC']))

        metrics = self._validation(thr)
        self.valid['LOSS'].append(metrics['LOSS'])
        self.valid['ACC'].append(metrics['ACC'])
        print('|>- Validation:\tLoss: {:.3f} \t Accuracy: {:.3f}'.format(metrics['LOSS'],
                                                                     metrics['ACC']))
        print('|')
        
    def train_model(self, epochs, thr):
        
        self.train = {'LOSS':[], 'ACC':[]}
        self.valid = {'LOSS':[], 'ACC':[]}
        
        plot_rate = 5
        
        for epoch in range(epochs):
            print('\n************************************')
            print('> Epoch {}/{}'.format(epoch, epochs-1))
            print('|')
            
            self._training_step(thr)
            
            # PLOTTING METRICS
            if(epoch % plot_rate == 0 and epoch!=0):
                plotMetrics(self)
                
    def kernel(self):
        
        return self.model.conv.weight
        
        
        
        
        
        
        
        