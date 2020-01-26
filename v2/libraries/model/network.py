#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:34:32 2019

@author: daniele
"""

#%% IMPORTS

#import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn.init import kaiming_uniform, kaiming_normal_
from libraries.torchsummary import summary
from torchvision import models
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
#    classname = mod.__class__.__name__
#    if classname.find('Conv') != -1:
#        xavier(mod.weight.data)
#        if(mod.bias is not None):
#            xavier(mod.bias.data)

    # XAVIER NORMAL INITIALIZATION
#    if isinstance(mod, nn.Conv2d):
#        xavier_norm(mod.weight.data)
#        if(mod.bias):
#            xavier_norm(mod.bias.data)
            
    # HE NORMAL INITIALIZATION --> better with relu / leaky_relu activations
    if isinstance(mod, nn.Conv2d):
        kaiming_normal_(mod.weight.data)
#        if(mod.bias):
#            kaiming_normal_(mod.bias.data)

#model.apply(weights_init)
        
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
    
class EncoderTL(nn.Module):
    
    def __init__(self, opt, z_size=None):
        super().__init__()
        
        if(z_size is None):
            z_size = opt.z_size
        else:
            z_size = z_size
        
        if(opt.tl == 'vgg16'):
            tl = models.vgg16(pretrained=True).cuda()
            
            for param in tl.parameters():
                param.require_grad = False
                
            modules = list(tl.children())[:-2][0]
            features = list(modules)[:-3]
            self.net = nn.Sequential(*features)
            self.net.add_module('Final conv2D', nn.Conv2d(512, z_size, 2))
            
            
        elif(opt.tl == 'resnet18'):
            tl = models.resnet18(pretrained=True).cuda()
            
            for param in tl.parameters():
                param.require_grad = False
                
#            modules = list(tl.children())[:-2]
#            net = nn.Sequential(*modules)
#            net.add_module('Final_Conv2D', nn.Conv2d(512, 100, 3, 1, 1))
            
            tl.fc = nn.Linear(512, 100)
            
            self.net = tl
                
#            self.net = net
        
    def forward(self, x):
        
        x = self.net(x)
        x = x.reshape(x.size(0), x.size(1), 1, 1)
        
        return x
        
        
        
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
    
    def __init__(self, opt, xavier_init=True):
        super().__init__()
        
        self.encoder1 = Encoder(opt)
        self.decoder  = Decoder(opt)
        self.encoder2 = Encoder(opt)
        
        # INITIALIZATION
        if(xavier_init):
            self.encoder1.apply(weights_init)
            self.decoder.apply(weights_init)
            self.encoder2.apply(weights_init)
        
    def forward(self, x):

        # LATENT REPRESENTATION
        z       = self.encoder1(x)
        
        # RECONSTRUCTED IMAGE
        x_prime = self.decoder(z)

        # RECONSTRUCTED LATENT REPRESENTATION
        z_prime = self.encoder2(x_prime)     
        
        return x_prime, z, z_prime
    
class GeneratorTL(nn.Module):
    
    def __init__(self, opt, xavier_init=True):
        super().__init__()
        
        self.encoder1 = EncoderTL(opt)
        self.decoder  = Decoder(opt)
        self.encoder2 = EncoderTL(opt)
        
        # INITIALIZATION
        if(xavier_init):
            self.encoder1.apply(weights_init)
            self.decoder.apply(weights_init)
            self.encoder2.apply(weights_init)
        
    def forward(self, x):

        # LATENT REPRESENTATION
        z       = self.encoder1(x)
        
        # RECONSTRUCTED IMAGE
        x_prime = self.decoder(z)

        # RECONSTRUCTED LATENT REPRESENTATION
        z_prime = self.encoder2(x_prime)     
        
        return x_prime, z, z_prime    
    
class Discriminator(nn.Module):
    
    def __init__(self, opt, xavier_init=True):
        super().__init__()
        
        model = Encoder(opt, 1)
        
        if(xavier_init):
            model.apply(weights_init)
        
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
        
def featureExtraction():

    vgg = models.vgg16_bn(pretrained=True)
    
    for param in vgg.parameters():
        param.requires_grad = False

    modules = list(vgg.children())[:-1]
    modules.append(nn.Sequential(nn.Conv2d(512, 100, 7)))

    features = nn.Sequential(*modules)

    return features   

def encoderFullyConv():
    encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 128, 4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 256, 4, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.Conv2d(256, 512, 4, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            
            nn.Conv2d(512, 100, (6,48)),
            
            )
    
    return encoder


def decoderFullyConv():
    
    decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, (4,25)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(1024, 512, (4,4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(512, 256, (4,4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(256, 128, (4,4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(128, 64, (4,4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(64, 32, (4,4), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(32, 3, (2,2), stride=2),
#
#            nn.Conv2d(3, 1, (256,1600)),
#            nn.Tanh()
            
            )
    
    return decoder

def fully_conv_layer_decoder():
    
    layer = nn.Sequential(
                nn.Conv2d(3, 1, (256,1600)),
                nn.Tanh())
    
    return layer

def final_layer_decoder():
    
    layer = nn.Sequential(
                nn.Tanh())
    
    return layer

class FCN_Generator(nn.Module):

    def __init__(self, xavier_init=True):

        super().__init__()
        
        self.encoder1 = featureExtraction()
        self.decoder = decoderFullyConv()               
        
        self.fullyConvLayer = fully_conv_layer_decoder()
        self.finalLayer = final_layer_decoder()
        
        self.encoder2 = encoderFullyConv()
        
        # INITIALIZATION
        if(xavier_init):
            self.encoder1.apply(weights_init)
            self.decoder.apply(weights_init)
            self.encoder2.apply(weights_init)
        
    def forward(self, x):
        
        z = self.encoder1(x)
        temp = self.decoder(z)
        
        x_conv = self.fullyConvLayer(temp)
        x_prime = self.finalLayer(temp)
        
        z_prime = self.encoder2(x_prime)
        
        return x_prime, z, z_prime, x_conv
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
