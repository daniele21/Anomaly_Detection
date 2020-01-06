# -*- coding: utf-8 -*-

#%%
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d

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

def gaussFilterScores(scores, sigma):
    
    try:
        scores = scores.cpu()
    except:
        scores = scores
    
    gauss_scores = gaussian_filter1d(scores, sigma=sigma)

    return gauss_scores

#def tune_KernelSize(model, mode='conv'):
#    '''
#        mode :  'conv' or 'median'
#
#    '''
#    
#    assert mode in ['conv', 'median'], 'Wrong mode input'
#
#    results = {'k':[], 'AUC':[], 'Thr':[]}
#    
#    scores = model.anomaly_scores
#    kernel_sizes  = np.arange(3,33,2)
#        
#    best = {'auc':0, 'k':0, 'thr':0}
#    
#    if(mode == 'median'):
#
#        for k in kernel_sizes:
#            
#            scores = medFilterScores(scores, k)
#            auc, thr = evaluate(model.gt_labels, scores)
#            
#            if(auc > best['auc']):
#                best['auc'] = auc
#                best['k'] = k
#                best['thr'] = thr
#                
#            results['k'].append(k)
#            results['AUC'].append(auc)
#            results['Thr'].append(thr)
#            
#    if(mode == 'conv'):
#
#        for k in kernel_sizes:
#            
#            scores = convFilterScores(scores, k)
#            auc, thr = evaluate(model.gt_labels, scores)
#            
#            if(auc > best['auc']):
#                best['auc'] = auc
#                best['k'] = k
#                best['thr'] = thr
#            
#            results['k'].append(k)
#            results['AUC'].append(auc)
#            results['Thr'].append(thr)
#    
#    __print_tuningResults(results, mode)
#    print('\n\n>____Best Option____\n')
#    print('> kernel_size: \t{}'.format(best['k']))
#    print('> auc        : \t{}'.format(best['auc']))
#    print('> threshold  : \t{}'.format(best['thr']))
#    
#    return best['k'], best['thr']
    
def tune_kernelSize(model, mode='conv'):
    '''
        mode :  'conv' or 'median' or 'gauss'

    '''
    
    assert mode in ['conv', 'median', 'gauss'], 'Wrong mode input'

    results = {'param':[], 'AUC':[], 'Thr':[]}
    
    kernel_sizes  = np.arange(3,33,2)
        
    best = {'auc':0, 'k':0, 'thr':0}

    for k in kernel_sizes:
        
        auc, thr = model.evaluateRoc(mode=mode, param=k)
        
        if(auc > best['auc']):
            best['auc'] = auc
            best['param'] = k
            best['thr'] = thr
            
        results['param'].append(k)
        results['AUC'].append(auc)
        results['Thr'].append(thr)

    __print_tuningResults(results, mode)
    print('\n\n_____Best Option____\n')
    print('> kernel_size: \t{}'.format(best['param']))
    print('> auc        : \t{}'.format(best['auc']))
    print('> threshold  : \t{}'.format(best['thr']))
    
    return best['param'], best['thr']

def tune_sigma(model, mode='gauss'):
    '''
        mode :  'conv' or 'median' or 'gauss'

    '''
    
    assert mode is not 'gauss', 'Wrong mode input'

    results = {'param':[], 'AUC':[], 'Thr':[]}
    
    sigmas  = np.arange(1,20,0.1)
        
    best = {'auc':0, 'param':0, 'thr':0}

    for s in sigmas:
        
        auc, thr = model.evaluateRoc(mode=mode, param=s)
        
        if(auc > best['auc']):
            best['auc'] = auc
            best['param'] = s
            best['thr'] = thr
            
        results['param'].append(s)
        results['AUC'].append(auc)
        results['Thr'].append(thr)

    __print_tuningResults(results, mode)
    print('\n\n_____Best Option____\n')
    print('> kernel_size: \t{}'.format(best['param']))
    print('> auc        : \t{}'.format(best['auc']))
    print('> threshold  : \t{}'.format(best['thr']))
    
    return best['param'], best['thr'] 
    
def __print_tuningResults(results, mode):
    
    print('\nResults Tuning {}'.format(mode))
    
    
    for i in range(len(results['param'])):
        print('\n')
        
        for x in ['param', 'AUC', 'Thr']:
            print(str(x) + ':\t\t{:.4f}'.format(results[x][i]))
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    