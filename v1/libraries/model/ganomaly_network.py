#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:02:01 2019

@author: daniele
"""
#%% IMPORTS
from libraries.model.network import Generator, Discriminator, weights_init
from libraries.model import loss
from libraries.model.options import Options

import torch
import torch.nn as nn

#%%
device = torch.device('cuda:0')


#%%
class GanomalyModel():
    
    def __init__(self, opt):
        
        self.generator = Generator(opt).to(device)
        self.discriminator = Discriminator(opt).to(device)
        
        # WEIGHT INITIALIZATION
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # LOSSES
        self.l_adv = loss.adversial_loss        
        self.l_con = loss.contextual_loss()
        self.l_enc = loss.encoder_loss
        self.l_bce = loss.binaryCrossEntropy_loss()
        self.gradLoss = loss.gradientLoss()
    
        # ADAPTIVE WEIGHT LOSSES
#        self.w_adv = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
#        self.w_con = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
#        self.w_enc = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
        
        self.w_adv = torch.FloatTensor([opt.w_adv])
        self.w_adv.requires_grad=True
        
        self.w_con = torch.FloatTensor([opt.w_con])
        self.w_con.requires_grad=True
        
        self.w_enc = torch.FloatTensor([opt.w_enc])
        self.w_enc.requires_grad=True
        
        
        self.params = [self.w_adv, self.w_con, self.w_enc]
        self.shared_layer = nn.Sequential(self.generator.decoder.net[-2:])
        self.alpha = opt.alpha
        
        # INIZIALIZATION INPUT TENSOR
        self.real_label = torch.ones (size=(opt.batch_size,), dtype=torch.float32, device=device)
        self.fake_label = torch.zeros(size=(opt.batch_size,), dtype=torch.float32, device=device)
    
    def init_optim(self, optim_gen, optim_discr, optimizer_weights):
        self.optimizer_gen = optim_gen
        self.optimizer_discr = optim_discr
        self.optimizer_weights = optimizer_weights
    
    def train(self):
        self.generator.train()
        self.discriminator.train()
        
    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()        
        
#    def loss_function_gen(self, x, x_prime, z, z_prime, feat_fake, feat_real, opt=None):
#        
#        if(opt is None):
#            # DEFAULT CASE
#            opt = Options()
#        
#        # LOSS GENERATOR
#        err_gen_adv = self.l_adv(feat_fake, feat_real)
#        err_gen_con = self.l_con(x_prime, x)
#        err_gen_enc = self.l_enc(z_prime, z)
#        
#        loss_gen = err_gen_adv * opt.w_adv + \
#                       err_gen_con * opt.w_con + \
#                       err_gen_enc * opt.w_enc
#                       
##        loss_gen_val = err_gen_con * opt.w_con + \
##                       err_gen_enc * opt.w_enc
#                       
#        return loss_gen, [err_gen_adv, err_gen_con, err_gen_enc]
        
    def loss_function_gen(self, x, x_prime, z, z_prime, feat_fake, feat_real, opt=None):
        
        if(opt is None):
            # DEFAULT CASE
            opt = Options()
        
        # LOSS GENERATOR
        adv_loss = self.l_adv(feat_fake, feat_real)
        con_loss = self.l_con(x_prime, x)
        enc_loss = self.l_enc(z_prime, z)
        
        self.w_adv_loss = self.params[0].cuda() * adv_loss
        self.w_con_loss = self.params[1].cuda() * con_loss
        self.w_enc_loss = self.params[2].cuda() * enc_loss
        
        loss_gen = self.w_adv_loss + self.w_con_loss + self.w_enc_loss
        
        return loss_gen, [adv_loss, con_loss, enc_loss]
                       
    def loss_function_discr(self, pred_real, pred_fake):
        
        # LOSS DISCRIMINATOR
        self.err_discr_real = self.l_bce(pred_real, self.real_label)
        self.err_discr_fake = self.l_bce(pred_fake, self.fake_label)
        
        self.loss_discr = 0.5 * (self.err_discr_real + self.err_discr_fake)
        
        return self.loss_discr
       
    # FORWARDS

    # GENERATOR
    def forward_gen(self, x):
        '''
            Forward propagate through GENERATOR
        
            x       : image
            x_prime : reconstructed image
            z       : latent vector
            z_prime : reconstructed latent vector
            
        '''   
        
        x_prime, z, z_prime = self.generator(x)
        
        return x_prime, z, z_prime
    
    # DISCRIMINATOR
    def forward_discr(self, x, x_prime):
        
        pred_real, feat_real = self.discriminator(x)
        pred_fake, feat_fake = self.discriminator(x_prime)
        
        return pred_real, feat_real, pred_fake, feat_fake

    # BACKWARDS
    
    def reInit_discr(self):
        
        self.discriminator.apply(weights_init)
        print('Reloding Weight init')
    
    def weighting_losses(self, l0):
    
        nTasks = len(self.params)
        param = list(self.shared_layer.parameters())
        
        G1R = torch.autograd.grad(self.w_adv_loss, param[0], retain_graph=True, create_graph=True)
        G1 = torch.norm(G1R[0], 2)
        
        G2R = torch.autograd.grad(self.w_con_loss, param[0], retain_graph=True, create_graph=True)
        G2 = torch.norm(G2R[0], 2) 
        
        G3R = torch.autograd.grad(self.w_enc_loss, param[0], retain_graph=True, create_graph=True)
        G3 = torch.norm(G3R[0], 2) 
        
        G_avg = torch.div((G1+G2+G3), nTasks)
        
        # Calculating relative losses 
        lhat1 = torch.div(self.w_adv_loss,l0)
        lhat2 = torch.div(self.w_con_loss,l0)
        lhat3 = torch.div(self.w_enc_loss,l0)

        lhat_avg = torch.div((lhat1+lhat2+lhat3), nTasks)
        
        # Calculating relative inverse training rates for tasks 
        inv_rate1 = torch.div(lhat1, lhat_avg)
        inv_rate2 = torch.div(lhat2, lhat_avg)
        inv_rate3 = torch.div(lhat3, lhat_avg)
        
        # Calculating the constant target for Eq. 2 in the GradNorm paper
        C1 = G_avg*(inv_rate1)**self.alpha
        C2 = G_avg*(inv_rate2)**self.alpha
        C3 = G_avg*(inv_rate3)**self.alpha
        C1 = C1.detach()
        C2 = C2.detach()
        C3 = C3.detach()
        
        self.optimizer_weights.zero_grad()
        
        # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
#        print('\n1.')
#        print(self.gradLoss(G1, C1))
#        print('2.')
#        print(self.gradLoss(G2, C2))
#        print('3.')
#        print(self.gradLoss(G3, C3))
        Lgrad = self.gradLoss(G1, C1) + self.gradLoss(G2, C2) + self.gradLoss(G3, C3)
        Lgrad.backward(retain_graph=True)
        
        # Updating loss weights
        self.optimizer_weights.step()
        
        
    def optimize_gen(self, loss_gen, l0):
        
        self.optimizer_gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        
        # ADAPTING WEIGHT LOSSES
        self.weighting_losses(l0)
        # --------------------
        
        # Updating the model weights
        self.optimizer_gen.step()
        
        # ADAPTING WEIGHT LOSSES
        
        # Renormalizing the losses weights
        coef = 3/(self.w_adv + self.w_con + self.w_enc)
        self.params = [coef*self.w_adv, coef*self.w_con, coef*self.w_enc]
        
        return self.w_adv, self.w_con, self.w_enc
        # --------------------
        
    
    def optimize_discr(self, loss_discr):
        
        self.optimizer_discr.zero_grad()
        loss_discr.backward()
        self.optimizer_discr.step()
        
        if(loss_discr.item() < 1e-5):
            self.reInit_discr()
            