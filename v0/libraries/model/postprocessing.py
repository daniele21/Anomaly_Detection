# -*- coding: utf-8 -*-

#%%
import numpy as np
from scipy.signal import medfilt

from libraries.model.evaluate import evaluate
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

def tune_KernelSize(model, mode='conv'):
    '''
        mode :  'conv' or 'median'

    '''
    
    assert mode in ['conv', 'median'], 'Wrong mode input'

    results = {'k':[], 'AUC':[], 'Thr':[]}
    
    scores = model.anomaly_scores
    kernel_sizes  = np.arange(3,23,2)
        
    if(mode == 'median'):

        for k in kernel_sizes:
            
            scores = medFilterScores(scores, k)
            auc, thr = evaluate(model.gt_labels, scores)
            
            results['k'].append(k)
            results['AUC'].append(auc)
            results['Thr'].append(thr)
            
    if(mode == 'conv'):

        for k in kernel_sizes:
            
            scores = convFilterScores(scores, k)
            auc, thr = evaluate(model.gt_labels, scores)
            
            results['k'].append(k)
            results['AUC'].append(auc)
            results['Thr'].append(thr)
    
    return results
    
