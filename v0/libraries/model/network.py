#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:34:32 2019

@author: daniele
"""

#%% IMPORTS

#import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_ as xavier
from torch.nn.init import kaiming_uniform as kaiming

from libraries.torchsummary import summary
#%% CONSTANTS

KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1

NEGATIVE_SLOPE = 0.2
#%%

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
#        mod.weight.data.normal_(0.0, 0.02)
        xavier(mod.weight.data)
        if(mod.bias is not None):
            xavier(mod.bias.data)
#    elif classname.find('BatchNorm') != -1:
#        mod.weight.data.normal_(1.0, 0.02)
#        mod.bias.data.fill_(0)
        
#%% NETWORK

class Encoder(nn.Module):
    '''
    DESCRIPTION:
        Encoder network for input data 32x32
    '''
    def __init__(self, opt, z_size=None):
        super().__init__()
        
        img_size = opt.img_size
        
        if(z_size is None):
            z_size = opt.z_size
        else:
            z_size = z_size
            
        in_channels = opt.in_channels
#        print('Encoder channels:')
#        print(in_channels)
        out_channels = opt.out_channels
        n_extra_layers = opt.n_extra_layers
        
        assert img_size % 16 == 0, "--> ERROR: image size is not divisible by 16"
        
        conv_img_size = img_size
        net = nn.Sequential()
        layer=1
        
        # LAYER 1
        
        # CONVOLUTION
        net.add_module('{}-Initial-Conv'.format(layer), nn.Conv2d(in_channels, out_channels,
                                                                   kernel_size=KERNEL_SIZE, stride=STRIDE,
                                                                   padding=PADDING, bias=False))
        # ACTIVATION FUNCTION
        net.add_module('{}-Initial-ReLu'.format(layer), nn.LeakyReLU(NEGATIVE_SLOPE, inplace=True))
        
        conv_img_size = img_size / 2
#        print('Conv Image size: ', conv_img_size)
        
        
        # EXTRA LAYERS
        for n_layer in range(n_extra_layers):
            layer += 1
             # CONVOLUTION
            net.add_module('{}-Extra-Conv'.format(layer), nn.Conv2d(out_channels, out_channels,
                                                                       kernel_size=3, stride=1,
                                                                       padding=PADDING, bias=False))
            # BATCH NORMALIZATION
            net.add_module('{}-Extra-Batch'.format(layer), nn.BatchNorm2d(out_channels))
            
            # ACTIVATION FUNCTION
            net.add_module('{}-Extra-ReLu'.format(layer), nn.LeakyReLU(NEGATIVE_SLOPE, inplace=True))

        
        
        # HIDDEN LAYERS
        while(conv_img_size > 4):
            layer += 1
            in_channels = out_channels
            out_channels = out_channels * 2
            
            # CONVOLUTION
            net.add_module('{}-Pyramid-Conv'.format(layer), nn.Conv2d(in_channels, out_channels,
                                                                       kernel_size=KERNEL_SIZE, stride=STRIDE,
                                                                       padding=PADDING, bias=False))
            # BATCH NORMALIZATION
            net.add_module('{}-Pyramid-Batch'.format(layer), nn.BatchNorm2d(out_channels))
            
            # ACTIVATION FUNCTION
            net.add_module('{}-Pyramid-ReLu'.format(layer), nn.LeakyReLU(NEGATIVE_SLOPE, inplace=True))

            conv_img_size = conv_img_size / 2
#            out_channels = out_channels * 2
#            print('Conv Image size: ', conv_img_size)
            
        
        # FINAL LAYER
        layer += 1
        net.add_module('{}-Final-Conv'.format(layer), nn.Conv2d(out_channels, z_size,
                                                                kernel_size=KERNEL_SIZE, stride=1,
                                                                padding=0, bias=False))
        
        self.net = net
        
    def forward(self, x):
        
       output = self.net(x)
       
       return output
    
   
#%%    
 
    
class Decoder(nn.Module):
    '''
    DESCRIPTION:
        Decoder network for input data 32x32
    '''
    def __init__(self, opt):
        super().__init__()
        
        img_size = opt.img_size
        z_size = opt.z_size
        in_channels = opt.in_channels
        out_channels = opt.out_channels
        n_extra_layers = opt.n_extra_layers
        out_enc = in_channels
        
        assert img_size % 16 == 0, '--> ERROR: image size is not multiple of 16'
        
        layer = 1
        conv_img_size = 4
        out_channels = out_channels // 2
        
        while(conv_img_size != img_size):
            conv_img_size = conv_img_size * 2
            out_channels  = out_channels * 2
        
        net = nn.Sequential()
        # LAYER 1
        
        # TRANSPOSE CONVOLUTION
        net.add_module('{}-Initial-ConvT'.format(layer), nn.ConvTranspose2d(z_size, out_channels, 
                                                                            kernel_size=KERNEL_SIZE,
                                                                            stride=1, padding=0, bias=False))
        # BATCH NORMALIZATION
        net.add_module('{}-Initial-Batch'.format(layer), nn.BatchNorm2d(out_channels))
        
        # ACTIVATION FUNCTION
        net.add_module('{}-Initial-ReLu'.format(layer), nn.LeakyReLU(True))
        
        
        # HIDDEN LAYERS
        
        conv_img_size = 4
        
        while(conv_img_size < img_size // 2):
            layer += 1
            in_channels = out_channels
            out_channels = out_channels // 2
            # TRANSPOSE CONVOLUTION
            net.add_module('{}-Pyramid-ConvT'.format(layer), nn.ConvTranspose2d(in_channels, out_channels,  
                                                                                kernel_size=KERNEL_SIZE,
                                                                                stride=STRIDE, padding=PADDING, bias=False))
            # BATCH NORMALIZATION
            net.add_module('{}-Pyramid-Batch'.format(layer), nn.BatchNorm2d(out_channels))
            
            # ACTVATION FUNCTION
            net.add_module('{}-Pyramid-ReLu'.format(layer), nn.LeakyReLU(True))
            
            conv_img_size *= 2
            
        
        # EXTRA LAYERS
        for n_layer in range(n_extra_layers):
            layer += 1
            # TRANSPOSE CONVOLUTION
            net.add_module('{}-Extra-ConvT'.format(layer), nn.ConvTranspose2d(out_channels, out_channels,  
                                                                                kernel_size=3,
                                                                                stride=1, padding=PADDING, bias=False))
            # BATCH NORMALIZATION
            net.add_module('{}-Extra-Batch'.format(layer), nn.BatchNorm2d(out_channels))
            
            # ACTVATION FUNCTION
            net.add_module('{}-Extra-ReLu'.format(layer), nn.LeakyReLU(True))
            
            
        # FINAL LAYER
        layer += 1
        in_channels = out_channels
        
        net.add_module('{}-Final-ConvT'.format(layer), nn.ConvTranspose2d(in_channels, out_enc, kernel_size=KERNEL_SIZE,
                                                               stride=STRIDE, padding=PADDING, bias=False))
        # ACTIVATION FUNCTION
        net.add_module('{}-Final-Tanh'.format(layer), nn.Tanh())
        
        self.net = net
#        print('out_channels: ', out_channels)
    
    def forward(self, x):
        
        output = self.net(x)
        
        return output
        
#%%
class Generator(nn.Module):
    
    def __init__(self, opt):
        super().__init__()
        
        self.encoder1 = Encoder(opt)
        
        self.decoder  = Decoder(opt)
        
        self.encoder2 = Encoder(opt)
        
    def forward(self, x):
        
#        print('--> Input to the generator: ', x.shape)
#        print(x.shape)

        # LATENT REPRESENTATION
        z       = self.encoder1(x)
        
        # RECONSTRUCTED IMAGE
        x_prime = self.decoder(z)

        # RECONSTRUCTED LATENT REPRESENTATION
        z_prime = self.encoder2(x_prime)     
        
        return x_prime, z, z_prime
    
class Discriminator(nn.Module):
    
    def __init__(self, opt):
        super().__init__()
        
        model = Encoder(opt, 1)
        
        layers = list(model.net.children())
        
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
