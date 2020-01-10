#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 19:30:04 2019

@author: daniele
"""

# -*- coding: utf-8 -*-

#%% IMPORTS
import time
from network import Generator, weights_init
from loss import contextual_loss
from utils import Paths
from utils import Checkpoint, saveInfo
from utils import ensure_folder
from utils import EarlyStopping
#import utils as ut

import torch
import torch.nn as nn
from torchvision import transforms as Transforms

from matplotlib import pyplot as plt
import numpy as np

from tqdm import tqdm
import sys
from torch.autograd import Variable
#import sys

paths = Paths()
#%% NETWORK

class GeneratorModel():
    
    def __init__(self, opt, optimizer, trainloader=None, validationloader=None):
        
        self.model      = Generator(opt).cuda()
        self.optimizer  = optimizer(self.model.parameters(), opt.lr)
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.opt = opt
    
    def loadModel(self, checkpoint):
        self.model = checkpoint.model
        self.optimizer = checkpoint.optimizer
    
    def loadTrainloader(self, trainloader):
        self.trainloader = trainloader
        
    def loadValidationLoader(self, validationloader):
        self.validationloader = validationloader
     
    def _trainOneEpoch(self, loss_function):
    
        trainLoss = []
        
        self.model.train()
        
        start = time.time()
        
        n = len(self.trainloader)
        for image, label in tqdm(self.trainloader, leave=True, position=0, total=n,
                                 file = sys.stdout, desc='Training\t'):
            
            tensor_image = torch.Tensor(image).cuda()
#            print(tensor_image.shape)
            
            # FORWARD
            x_prime, z, z_prime = self.model(tensor_image)        
#            output.shape                    
            loss = loss_function(x_prime, tensor_image)
                    
            # BACKWARD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # VISUALIZATION
#            self.loss['train'].append(loss.item())
            trainLoss.append(loss.item())
            
        time_spent = time.time() - start
            
        return trainLoss, time_spent
        
    def _validation(self, loss_function):
    
        validationLoss = []
    
        self.model.eval()      
        
        start = time.time()
        
        with torch.no_grad():
            count=0
            self.curr_steps += self.batch_size
            
            n = len(self.validationloader)
            
            for image, label in tqdm(self.validationloader, leave=True, position=0, total=n,
                                     file = sys.stdout, desc='Validation\t'):
                
                count += 1
                
                img_var = Variable(image).cuda()
                img_var.shape
                
                # FORWARD
                x_prime, z, z_prime = self.model(img_var)        
#                output.shape
                        
                loss = loss_function(x_prime, img_var)
                
                # VISUALIZATION
#                self.loss['validation'].append(loss.item())
                validationLoss.append(loss.item())
                
            
            time_spent = time.time() - start
            
        return validationLoss, time_spent
        
    
    def train_autoencoder(self, save=True):
        
        self.curr_steps = 0
        self.printing_freq = self.opt.printing_freq
        self.batch_size = self.opt.batch_size
        
        es = EarlyStopping(self.opt)
        
        loss_function = contextual_loss()
        
        self.loss = {}
        self.loss['train'] = []
        self.loss['validation'] = []
        
        self.avg_loss = {}
        self.avg_loss['train'] = []
        self.avg_loss['validation'] = []
        
        for self.epoch in range(self.opt.epochs):
            print('\n')
            print('Epoch {}/{}'.format(self.epoch+1, self.opt.epochs))
            
            # TRAINING
            train_losses, train_time = self._trainOneEpoch(loss_function)
            train_loss = np.average(train_losses)
#            self.loss['train'].append(train_losses) 
            self.avg_loss['train'].append(train_loss)
            
            # VALIDATION
            val_losses, val_time     = self._validation( loss_function)
            val_loss = np.average(val_losses)
#            self.loss['validation'].append(val_losses)
            self.avg_loss['validation'].append(val_loss)
            
            print('\n')
            print('>- Training Loss:   {:.4f} in {:.2f} sec'.format(train_loss, train_time) )
            print('>- Validation Loss: {:.4f} in {:.2f} sec'.format(val_loss, val_time))
            
            # SAVING CHECKPOINT
            folder_save = paths.checkpoint_folder
            ensure_folder(folder_save)
            
#            filename = '{0}/{1}_lr:{2}|Epoch:{3}|Loss:{4:.3f}.pth.tar'.format(folder_save,
#                                                                                self.opt.name,
#                                                                                self.opt.lr,
#                                                                                epoch, train_loss)
            
            save = es(val_loss)
            if(save):
                self.saveCheckPoint(val_loss)
                self.plotting()
            
            if(es.early_stop):
                print('-> Early stopping now')
#                self.plotting()
                break
            
        self.plotting()
        
        self.saveInfo()
        
#        return val_loss
    
    def saveInfo(self):
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        saveInfo(self.opt, folder_save)
        
#        return val_loss
        
    def saveCheckPoint(self, val_loss):
        
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        path_file = '{0}/{1}_lr:{2}|Epoch:{3}|Loss:{4:.4f}.pth.tar'.format(folder_save,
                                                                             self.opt.name,
                                                                             self.opt.lr,
                                                                             self.epoch,
                                                                             val_loss)
        
        torch.save(self.model, path_file) 
        
    def plotting(self):
                
        # PLOTTING LOSSES
        plt.title('Average Loss')
        plt.plot(self.avg_loss['train'], color='r', label='train')
        plt.plot(self.avg_loss['validation'], color='b', label='validation')
        plt.legend()
        plt.show()
        
#        plt.savefig(paths.checkpoint_folder + self.opt.name + '_lr:{}.png'.format(self.opt.lr))
#        
    def predict(self, image):
        image_tensor = torch.FloatTensor(image)
        image_tensor = Transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze_(0)
        image_var = Variable(image_tensor).cuda()
        
#        plt.imshow(image)

        with torch.no_grad():
            x_prime, z, z_prime = self.model(image_var)

        output = x_prime.cpu().numpy()
        output = np.transpose(output[0], (2,1,0))
        
        final_output = (output * 0.5) / 0.5
        final_output = np.flip(final_output, 1)
        final_output = np.rot90(final_output, 1)        
        
#        plt.imshow(final_output)

        fig, [ax1, ax2] = plt.subplots(2,1, figsize=(15,15))
        ax1.imshow(image)
        ax2.imshow(final_output)
        
    def tuneLearningRate(self):
        
        max_count = 10
        result = []
        
        for count in range(max_count):
            
            self.opt.epochs = 5
            self.optimizer.lr = 10**np.random.uniform(-3, -6)
            loss = self.train_autoencoder(save=False)
            
            result.append(str(self.optimizer.lr) + ' : ' + str(loss))
        
        return result
                       
        
        
        
        











