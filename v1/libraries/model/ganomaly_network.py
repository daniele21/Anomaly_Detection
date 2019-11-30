#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:02:01 2019

@author: daniele
"""
#%% IMPORTS
from libraries.model.network import Generator, Discriminator, weights_init
from libraries.model.loss import adversial_loss, contextual_loss, encoder_loss, binaryCrossEntropy_loss
from libraries.model.options import Options

import torch

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
        self.l_adv = adversial_loss        
        self.l_con = contextual_loss()
        self.l_enc = encoder_loss
        self.l_bce = binaryCrossEntropy_loss()
        
        # INIZIALIZATION INPUT TENSOR
        self.real_label = torch.ones (size=(opt.batch_size,), dtype=torch.float32, device=device)
        self.fake_label = torch.zeros(size=(opt.batch_size,), dtype=torch.float32, device=device)
    
    def init_optim(self, optim_gen, optim_discr):
        self.optimizer_gen = optim_gen
        self.optimizer_discr = optim_discr
    
    def train(self):
        self.generator.train()
        self.discriminator.train()
        
    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()        
        
    def loss_function_gen(self, x, x_prime, z, z_prime, feat_fake, feat_real, opt=None):
        
        if(opt is None):
            # DEFAULT CASE
            opt = Options()
        
        # LOSS GENERATOR
        err_gen_adv = self.l_adv(feat_fake, feat_real)
        err_gen_con = self.l_con(x_prime, x)
        err_gen_enc = self.l_enc(z_prime, z)
        
        loss_gen = err_gen_adv * opt.w_adv + \
                       err_gen_con * opt.w_con + \
                       err_gen_enc * opt.w_enc
                       
#        loss_gen_val = err_gen_con * opt.w_con + \
#                       err_gen_enc * opt.w_enc
                       
        return loss_gen, [err_gen_adv, err_gen_con, err_gen_enc]
                       
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
    
    def optimize_gen(self, loss_gen):
        
        self.optimizer_gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        self.optimizer_gen.step()
    
    def optimize_discr(self, loss_discr):
        
        self.optimizer_discr.zero_grad()
        loss_discr.backward()
        self.optimizer_discr.step()
        
        if(loss_discr.item() < 1e-5):
            self.reInit_discr()
            