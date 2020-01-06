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
    
def tune_KernelSize(model, mode='conv'):
    '''
        mode :  'conv' or 'median'

    '''
    
    assert mode in ['conv', 'median'], 'Wrong mode input'

    results = {'k':[], 'AUC':[], 'Thr':[]}
    
    kernel_sizes  = np.arange(3,33,2)
        
    best = {'auc':0, 'k':0, 'thr':0}

    for k in kernel_sizes:
        
        auc, thr = model.evaluateRoc(mode=mode, kernel_size=k, plot=False)
        
        if(auc > best['auc']):
            best['auc'] = auc
            best['k'] = k
            best['thr'] = thr
            
        results['k'].append(k)
        results['AUC'].append(auc)
        results['Thr'].append(thr)

    __print_tuningResults(results, mode)
    print('\n\n>____Best Option____\n')
    print('> kernel_size: \t{}'.format(best['k']))
    print('> auc        : \t{}'.format(best['auc']))
    print('> threshold  : \t{}'.format(best['thr']))
    
    return best['k'], best['thr']
    
def __print_tuningResults(results, mode):
    
    print('\nResults Tuning {}'.format(mode))
    
    
    for i in range(len(results['k'])):
        print('\n')
        
        for x in ['k', 'AUC', 'Thr']:
            print(str(x) + ':\t\t{:.4f}'.format(results[x][i]))
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    