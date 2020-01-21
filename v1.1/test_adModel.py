#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:47:00 2019

@author: daniele
"""
#%% IMPORTS
from libraries.model.adModel import AnomalyDetectionModel
from libraries.model import evaluate

import torch
import unittest

#%%
class test_adModel(unittest.TestCase):
    
    def test_auc(self):
        model_path = '../../ckp_v1/Ganom_1_prova.1/Ganom_1_prova.1_lr:0.0001|Epoch:0|Auc:0.639|Loss:415.6041.pth.tar'
        adModel = torch.load(model_path)
        
        device = torch.device('cuda:0')
        test = adModel.testloader
        
        i = 0
        anomaly_scores = torch.zeros(size=(len(test.dataset),), dtype=torch.float32, device=device)
        gt_labels = torch.zeros(size=(len(test.dataset),), dtype=torch.long,    device=device)
        
        for images, labels in adModel.testloader:
            
            x = torch.Tensor(images).cuda()
            tensor_labels = torch.Tensor(labels).cuda()                
            
            _, z, z_prime = adModel.model.forward_gen(x)
            
            # ANOMALY SCORE
            score = torch.mean(torch.pow((z-z_prime), 2), dim=1)                
            
            anomaly_scores[i*64 : i*64 + score.size(0)] = score.reshape(score.size(0))
            gt_labels[i*64 : i*64 + score.size(0)] = tensor_labels.reshape(score.size(0))

        anomaly_scores_norm = (anomaly_scores - torch.min(anomaly_scores)) / (torch.max(anomaly_scores) - torch.min(anomaly_scores))
        auc, threshold_norm = evaluate(gt_labels, anomaly_scores_norm)

#unittest.main()
        
#%%
test = test_adModel()
test.test_auc()
        
#%%
model_path = '../../ckp_v1/Ganom_1_prova.1/Ganom_1_prova.1_lr:0.0001|Epoch:0|Auc:0.639|Loss:415.6041.pth.tar'
adModel = torch.load(model_path)

device = torch.device('cuda:0')
test = adModel.testloader

i = 0
anomaly_scores = torch.zeros(size=(len(test.dataset),), dtype=torch.float32, device=device)
gt_labels = torch.zeros(size=(len(test.dataset),), dtype=torch.long,    device=device)

with torch.no_grad():
    
    for images, labels in test:
        print(i)
        x = torch.Tensor(images).cuda()
        tensor_labels = torch.Tensor(labels).cuda()                
        
        _, z, z_prime = adModel.model.forward_gen(x)
        
        # ANOMALY SCORE
        score = torch.mean(torch.pow((z-z_prime), 2), dim=1)                
        
        anomaly_scores[i*adModel.opt.batch_size : i*adModel.opt.batch_size + score.size(0)] = score.reshape(score.size(0))
        gt_labels[i*adModel.opt.batch_size : i*adModel.opt.batch_size + score.size(0)] = tensor_labels.reshape(score.size(0))
    
        i += 1

print('Finish')
anomaly_scores_norm = (anomaly_scores - torch.min(anomaly_scores)) / (torch.max(anomaly_scores) - torch.min(anomaly_scores))
auc, threshold_norm = evaluate.evaluate(gt_labels, anomaly_scores_norm, plot=True)
#%%
from sklearn.metrics import roc_curve, auc, average_precision_score, recall_score
len(anomaly_scores_norm)
anomaly_scores_norm[-50:]

roc_curve(gt_labels, anomaly_scores_norm)




