# -*- coding: utf-8 -*-

#%%
import numpy as np

from scipy.signal import medfilt

#%% FUNCTIONS

def convFilterScores(scores, kernel_size):
    
    try:
        scores = scores.cpu()
    except:
        scores = scores
        
    kernel = np.ones(kernel_size) / kernel_size
    
    conv_scores = np.convolve(scores, kernel, mode='same')
        
    return conv_scores


def medFilterScores(scores, kernel_size):
    
    try:
        scores = scores.cpu()
    except:
        scores = scores
    
    med_scores = medfilt(scores, kernel_size=kernel_size)

    return med_scores

def postProcessing(kernel_size, mode='conv'):
    '''
        mode: 'conv' or 'median'
    '''
    
    
