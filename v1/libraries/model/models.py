#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:56:49 2019

@author: daniele
"""
#%% IMPORTS
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchvision import models as pre_models

from libraries.model.evaluate import evaluate

device = torch.device('cuda:0') 
#%% 

class BinClassifierModel():
    
    def __init__(self, model, optimizer, dataloader):
        
        self.model = model.cuda()
#        self.model = CNN2().cuda()
        self.trainloader = dataloader['train']
        self.validloader = dataloader['validation']
        self.testloader = dataloader['test']
        
        self.loss_function = nn.BCELoss()
        self.optimizer = optimizer
        
    def _train_one_epoch(self):
        
        losses = []
        correct = 0
        total = 0
        
        for images, labels in self.trainloader:
            
            x = torch.Tensor(images).cuda()
            labels = torch.Tensor(labels).cuda()
            
            output = self.model(x)
            output = output.reshape(-1)
            loss = self.loss_function(output, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            losses.append(loss.item() * x.size(0))
                
            # ACCURACY            
            predictions = np.round(output.detach().cpu())
            targets = np.round(labels.detach().cpu())
            
            correct += (predictions == targets).sum().item()
            total += labels.size(0)
            
        loss_value = np.mean(losses)
        accuracy_value = correct/total
        
        return {'LOSS':loss_value, 'ACC':accuracy_value}
    
    def _validation(self):
        
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
                
            for images, labels in self.validloader:
                
                x = torch.Tensor(images).cuda()
                labels = torch.Tensor(labels).cuda()
                
                output = self.model(x)
                output = output.reshape(-1)
                loss = self.loss_function(output, labels)
                
                predictions = np.round(output.cpu())
                targets = np.round(labels.cpu())
                
                correct += (predictions == targets).sum().item()
                total += labels.size(0)
            
                losses.append(loss.item() * x.size(0))
                
            loss_value = np.mean(losses)
            accuracy_value = correct/total
        
        return {'LOSS':loss_value, 'ACC':accuracy_value}
        
    def train_model(self, epochs):
        
        self.train = {'LOSS':[], 'ACC':[]}
        self.valid = {'LOSS':[], 'ACC':[]}
        
        plot_rate = 5
        
        for epoch in range(epochs):
            print('\n************************************')
            print('> Epoch {}/{}'.format(epoch, epochs-1))
            print('|')
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
            
            # PLOTTING METRICS
            if(epoch % plot_rate == 0 and epoch!=0):
                self.plotMetrics()
            
            
    
    def inference(self, dataloader, i):
        
        image = dataloader.dataset.data[i]
        imageTensor, labelTensor = dataloader.dataset.__getitem__(i)
        
        x = imageTensor.unsqueeze(0).cuda()
        
        with torch.no_grad():
            output = self.model(x)
            print(output.shape)
            print(output[:,:,:,0][0])
            
        pred = np.round(output.cpu().item())
        
        plt.title('Target: {}\nPred.:   {}'.format(labelTensor.item(), pred))
        plt.imshow(image)
       
#        plt.imshow(output[:,:,:,0][0])
#        output
        
        return output
            
    def plotMetrics(self):
        fig, [ax1, ax2] = plt.subplots(2,1, figsize=(6,10))
        _subplot(ax1, self.train['LOSS'], self.valid['LOSS'], 'Loss')
        _subplot(ax2, self.train['ACC'], self.valid['ACC'], 'Accuracy')
        plt.show()
        
#%%
class FCNmodel():
    
    def __init__(self, model, optimizer, dataloader):
        
        self.model = model.cuda()
#        self.model = CNN2().cuda()
        self.trainloader = dataloader['train']
        self.validloader = dataloader['validation']
        self.testloader = dataloader['test']
        
        self.loss_function = nn.BCELoss()
        self.optimizer = optimizer
        
    def _train_one_epoch(self):
        
        losses = []
        correct = 0
        total = 0
        i=0
        for images, labels in self.trainloader:
#            print(i)
            x = torch.Tensor(images).cuda()
            # Flattening the mask target
            labels = labels.reshape(-1)
            labels = torch.Tensor(labels.float()).cuda()

            output = self.model(x)
            
            # flattening output
            output = output.reshape(-1)
            output = torch.sigmoid(output)
#            print('max out: {}'.format(torch.max(output)))
#            print('max lab: {}'.format(torch.max(labels)))
#            print('min out: {}'.format(torch.min(output)))
#            print('min lab: {}'.format(torch.min(labels)))
            
            loss = self.loss_function(output, labels)
#            print(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            losses.append(loss.item() * x.size(0))
                
            # ACCURACY            
            predictions = np.round(output.detach().cpu())
            targets = np.round(labels.detach().cpu())
            
#            print('> output')
#            print(predictions)
#            print('> labels')
#            print(targets)
            
            correct += (predictions == targets).sum().item()
            total += labels.size(0)
#            print('> Accuracy')
#            print(correct/total)
            i +=1
        loss_value = np.mean(losses)
        accuracy_value = correct/total
#        exit()
        return {'LOSS':loss_value, 'ACC':accuracy_value}
    
    def _validation(self):
        
        losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
                
            for images, labels in self.validloader:
                
                x = torch.Tensor(images).cuda()
                labels = labels.reshape(-1)
                labels = torch.Tensor(labels.float()).cuda()
                
                output = self.model(x)
                output = output.reshape(-1)
                output = torch.sigmoid(output)
                
                loss = self.loss_function(output, labels)
                
                predictions = np.round(output.cpu())
                targets = np.round(labels.cpu())
                
                # ACCURACY
                correct += (predictions == targets).sum().item()
                total += labels.size(0)
#                print('> Accuracy')
#                print(correct/total)
            
                losses.append(loss.item() * x.size(0))
                
            loss_value = np.mean(losses)
            accuracy_value = correct/total
        
        return {'LOSS':loss_value, 'ACC':accuracy_value}
    
    def _test(self):
        
        losses = []
        correct = 0
        total = 0
        i = 0
        
        with torch.no_grad():
               
            predictions = torch.zeros(size=(len(self.testloader.dataset),), dtype=torch.float32, device=device)
            targets = torch.zeros(size=(len(self.testloader.dataset),), dtype=torch.long,    device=device)
            
            for images, labels in self.testloader:
                
                x = torch.Tensor(images).cuda()
#                labels = labels.reshape(-1)
                labels = torch.Tensor(labels.float()).cuda()
                
                output = self.model(x)
#                output = output.reshape(-1)
                output = torch.sigmoid(output)
                
                prediction = np.round(output.cpu())
                target = np.round(labels.cpu())
                
                # ACCURACY
                correct += (prediction == target).sum().item()
                total += labels.size(0)
                
                batch_size = images.shape[0]
                print(prediction.shape)
#                print(prediction.reshape(prediction.size(2), prediction.size(3)))
                predictions.expand(prediction)
                targets[i*batch_size : i*batch_size + target.size(0)] = target.reshape(target.size(0))
                
                i += 1
                
            accuracy_value = correct/total
            
            # NORMALIZING - scaling values between 0 and 1 -- [0, 1]
            auc, thr_norm = evaluate(labels, predictions, plot=True)
            
        return {'LOSS':loss_value, 'ACC':accuracy_value}
        
    def train_model(self, epochs):
        
        self.train = {'LOSS':[], 'ACC':[]}
        self.valid = {'LOSS':[], 'ACC':[]}
        
        plot_rate = 5
        
        for epoch in range(epochs):
            print('\n************************************')
            print('> Epoch {}/{}'.format(epoch, epochs-1))
            print('|')
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
            
            metrics = self._test()
            print('|')
            
            # PLOTTING METRICS
            if(epoch % plot_rate == 0 and epoch!=0):
                self.plotMetrics()
            
            
    
    def inference(self, dataloader, i):
        
        image = dataloader.dataset.data[i]
        imageTensor, labelTensor = dataloader.dataset.__getitem__(i)
        
        x = imageTensor.unsqueeze(0).cuda()
        
        with torch.no_grad():
            output = self.model(x)
            
            output = torch.sigmoid(output)
#            print(output.shape)
#            print(output[:,:,:,0][0])
            
        pred = np.round(output.cpu())
        print('MAX: {}'.format(torch.max(pred))) 
        print(pred[0][0])
        
#        plt.title('Target: {}\nPred.:   {}'.format(labelTensor.item(), pred))
        plt.imshow(image)
        plt.show()
        plt.imshow(pred[0][0])
        plt.show()
#        plt.imshow(output[:,:,:,0][0])
#        output
        
        return image, output
           
    def testInference(self, dataloader):
        
        count_ones = 0
        
        for i in range(len(dataloader.dataset.data)):
#            print(i)
            imageTensor, labelTensor = dataloader.dataset.__getitem__(i)
            
            x = imageTensor.unsqueeze(0).cuda()
            
            with torch.no_grad():
                output = self.model(x)
                
                output = torch.sigmoid(output)
    #            print(output.shape)
    #            print(output[:,:,:,0][0])
                
            pred = np.round(output.cpu())
            max_value = torch.max(pred)
    #        print('MAX: {}'.format(torch.max(pred)))
            if(max_value != 0):
                print('{} ONE'.format(i))
                count_ones += 1
                
        return count_ones
    
    def plotMetrics(self):
        fig, [ax1, ax2] = plt.subplots(2,1, figsize=(6,10))
        _subplot(ax1, self.train['LOSS'], self.valid['LOSS'], 'Loss')
        _subplot(ax2, self.train['ACC'], self.valid['ACC'], 'Accuracy')
        plt.show()
#%%
def _subplot(ax, train, val, title):
    ax.set_title(title)
    ax.plot(train, color='r', label='Training')
    ax.plot(val, color='b', label='Validation')
    ax.legend()
      
