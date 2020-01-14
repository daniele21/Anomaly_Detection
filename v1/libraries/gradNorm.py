#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 00:06:41 2019

@author: daniele
"""

#%% IMPORTS
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from libraries.model.network import Encoder, Decoder
#%%

class MultiTaskLoss(nn.Module):
    
    def __init__(self, opt):
        super().__init__()
        
        self.encoder_enc = Encoder(opt)
        self.decoder_con = Decoder(opt)
        
        decoder_layers = list(self.decoder_con.net.children())
        self.shared_layer = nn.Sequential(*decoder_layers[:-1])
        
        encoder = Encoder(opt, 1)
        layers_adv = list(encoder.net.children())
        self.encoder_adv = nn.Sequential(*layers_adv[:-1])
        
        
    def forward(self, x):
        # OUTPUT DECODER OF GENERATOR
        x_prime = self.shared_layer(x)
        
        # OUTPUT LAST ENCODER
        z_prime = self.encoder_enc(x_prime)
        
        # OUTPUTS ADVERSARIAL ENCODER
        feat_real = self.encoder_adv(x)
        feat_fake = self.encoder_adv(x_prime)
        
        return x_prime, z_prime, feat_real, feat_fake

def random_mini_batches(XE, R1E, R2E, mini_batch_size = 10, seed = 42): 
    # Creating the mini-batches
    np.random.seed(seed)            
    m = XE.shape[0]                  
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_XE = XE[permutation,:]
    shuffled_X1R = R1E[permutation]
    shuffled_X2R = R2E[permutation]
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_XE = shuffled_XE[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch_X1R = shuffled_X1R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X2R = shuffled_X2R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R)
        mini_batches.append(mini_batch)
    Lower = int(num_complete_minibatches * mini_batch_size)
    Upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))
    if m % mini_batch_size != 0:
        mini_batch_XE = shuffled_XE[Lower : Lower + Upper, :]
        mini_batch_X1R = shuffled_X1R[Lower : Lower + Upper]
        mini_batch_X2R = shuffled_X2R[Lower : Lower + Upper]
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R)
        mini_batches.append(mini_batch)
    
    return mini_batches

Weightloss1 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
Weightloss2 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)

params = [Weightloss1, Weightloss2]
nTasks = len(params)
MTL = MultiTaskLoss()
opt1 = torch.optim.Adam(MTL.parameters(), lr=LR)
opt2 = torch.optim.Adam(params, lr=LR)

loss_func = nn.MSELoss()
Gradloss = nn.L1Loss()


alph = 0.16
for it in range(epoch):
    epoch_cost = 0
    epoch_cost1 = 0
    epoch_cost2 = 0
    coef = 0
    num_minibatches = int(input_size / mb_size) 
    minibatches = random_mini_batches(X_train, Y1_train, Y2_train, mb_size)
    for minibatch in minibatches:
        XE, YE1, YE2  = minibatch 
        
        Yhat1, Yhat2 = MTL(XE)
        
        # l1 = w1 * loss_func1
        l1 = params[0]*loss_func(Yhat1, YE1.view(-1,1))    
        # l2 = w2 * loss_func2
        l2 = params[1]*loss_func(Yhat2, YE2.view(-1,1))
        
#        loss = torch.div(torch.add(l1,l2), 2)
        loss = torch.add(l1,l2)

        # for the first epoch with no l0
        if it == 0:
            l0 = loss.data        
        
        opt1.zero_grad()
        
        loss.backward(retain_graph=True)   
        
        # Getting gradients of the first layers of each tower and calculate their l2-norm 
        
        param = list(MTL.parameters())
        # ------------------(param[0])----------------
        G1R = torch.autograd.grad(l1, param[0], retain_graph=True, create_graph=True)
        G1 = torch.norm(G1R[0], 2)
        
        # ------------------(param[0])----------------
        G2R = torch.autograd.grad(l2, param[0], retain_graph=True, create_graph=True)
        G2 = torch.norm(G2R[0], 2)
        
#        G3R = torch.autograd.grad(l2, param[0], retain_graph=True, create_graph=True)
#        G3 = torch.norm(G2R[0], 2)
        
        G_avg = torch.div(torch.add(G1, G2), nTasks)
        
        # Calculating relative losses 
        lhat1 = torch.div(l1,l0)
        lhat2 = torch.div(l2,l0)
#        lhat3
        lhat_avg = torch.div(torch.add(lhat1, lhat2), nTasks)
        
        # Calculating relative inverse training rates for tasks 
        inv_rate1 = torch.div(lhat1,lhat_avg)
        inv_rate2 = torch.div(lhat2,lhat_avg)
#        inv_rate3
        
        # Calculating the constant target for Eq. 2 in the GradNorm paper
        C1 = G_avg*(inv_rate1)**alph
        C2 = G_avg*(inv_rate2)**alph
        C1 = C1.detach()
        C2 = C2.detach()
        
        opt2.zero_grad()
        # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
        Lgrad = torch.add(Gradloss(G1, C1),Gradloss(G2, C2))
        Lgrad.backward()
        
        # Updating loss weights 
        opt2.step()

        # Updating the model weights
        opt1.step()

        # Renormalizing the losses weights
        coef = 2/torch.add(Weightloss1, Weightloss2)
        params = [coef*Weightloss1, coef*Weightloss2]
        #print("Weights are:",Weightloss1, Weightloss2)
        #print("params are:", params)
        epoch_cost = epoch_cost + (loss / num_minibatches)
        epoch_cost1 = epoch_cost1 + (l1 / num_minibatches)
        epoch_cost2 = epoch_cost2 + (l2 / num_minibatches)
        
    costtr.append(torch.mean(epoch_cost))
    cost1tr.append(torch.mean(epoch_cost1))
    cost2tr.append(torch.mean(epoch_cost2))
    
    with torch.no_grad():
        Yhat1D, Yhat2D = MTL(X_valid)
        l1D = params[0]*loss_func(Yhat1D, Y1_valid.view(-1,1))
        l2D = params[1]*loss_func(Yhat2D, Y2_valid.view(-1,1))
        cost1D.append(l1D)
        cost2D.append(l2D)
        costD.append(torch.add(l1D,l2D)/2)
        print('Iter-{}; MTL loss: {:.4}'.format(it, loss.item()))
        #print('Iter-{}; Grad loss: {:.4}'.format(it, Lgrad.item()))
    
plt.plot(np.squeeze(costtr),'-r',np.squeeze(costD), '-b')
plt.ylabel('total cost')
plt.xlabel('iterations')
plt.show() 

plt.plot(np.squeeze(cost1tr),'-r', np.squeeze(cost1D), '-b')
plt.ylabel('task 1 cost')
plt.xlabel('iterations')
plt.show() 

plt.plot(np.squeeze(cost2tr),'-r', np.squeeze(cost2D),'-b')
plt.ylabel('task 2 cost')
plt.xlabel('iterations')
plt.show()



