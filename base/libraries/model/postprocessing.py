# -*- coding: utf-8 -*-

#%%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

from scipy.stats import norm, gaussian_kde
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d

from libraries.model.evaluate import evaluate
from libraries.utils import ensure_folder
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

    
def tune_kernelSize(model, mode='conv'):
    '''
        mode :  'conv' or 'median' or 'gauss'

    '''
    
    assert mode in ['conv', 'median', 'gauss'], 'Wrong mode input'

    results = {'param':[], 'AUC':[], 'Thr':[]}
    
    kernel_sizes  = np.arange(3,33,2)
        
    best = {'auc':0, 'k':0, 'thr':0}

    for k in kernel_sizes:
        
        auc, thr = model.evaluateRoc(mode=mode, param=k, plot=False)
        
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
    
    assert mode is 'gauss', 'Wrong mode input'

    results = {'param':[], 'AUC':[], 'Thr':[]}
    
    sigmas  = np.arange(1,20,0.1)
        
    best = {'auc':0, 'param':0, 'thr':0}

    for s in sigmas:
        
        auc, thr = model.evaluateRoc(mode=mode, param=s, plot=False)
        
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
            
    
def distScores(anomaly_scores, gt_labels, performance, figsize=(10,6),
               folder_save=None):
    
    try:
        anomaly_scores = anomaly_scores.cpu()
        gt_labels = gt_labels.cpu()
    except:
        anomaly_scores = anomaly_scores
        gt_labels = gt_labels
    
    anom_indexes = np.where(gt_labels==1)[0]
    normal_indexes = np.where(gt_labels==0)[0]
    
    # CHECK
    for item in anom_indexes:
        assert item not in normal_indexes, 'Anomaly in Normal set'   
    # END CHECK
    
    anomalies = [anomaly_scores[i] for i in anom_indexes]
    normals = [anomaly_scores[i] for i in normal_indexes]


    # PLOTTING
    plt.figure(figsize=figsize)
    x_limit = np.mean(anomalies)
    
    values, _, _ = plt.hist([anomalies, normals], bins=1000,
             label=['Anomaly Scores', 'Normal Scores'], density=True)

#    x1, y1 = [threshold, threshold], [0, max(values[1])]
#    plt.plot(x1, y1, marker='o', c='r', label='Threshold')
    
    __plottingThresholds(performance, max(values[1]))
    
    plt.xlim(0, x_limit)
    plt.legend(loc='best')
    plt.xlabel('Score')
    plt.show()
    
    plt.figure(figsize=figsize)
    # THRESHOLD
    __plottingThresholds(performance, max(values[1]))
    
    # DISTRIBUTIONS 
    
#    sn.distplot(anomalies, bins=1000, norm_hist=True, label='Anomaly Score')
#    sn.distplot(normals, bins=1000, norm_hist=True, label='Normal Score')
    
    sn.distplot(anomalies, bins=1000, kde=False, hist=True, norm_hist=True, label='Anomaly Score')
    sn.distplot(normals, bins=1000, kde=False, hist=True, norm_hist=True,label='Normal Score')

    plt.xlim(0, x_limit)
    plt.legend()
    
    if(folder_save is not None):
        ensure_folder(folder_save)
        print('> Saving Distribution Score at .. {}'.format(folder_save))
        plt.savefig(folder_save + 'distribution.png')
    
    plt.xlabel('Score')
    plt.show()
    
    
def __plottingThresholds(performance, h):
    
    thresholds = {'standard' : performance['standard']['Threshold'],
                  'conv' : performance['conv']['Threshold'],
                  'median' : performance['median']['Threshold'],
                  'gauss' : performance['gauss']['Threshold']}
    
    aucs = {'standard' : performance['standard']['AUC'],
              'conv' : performance['conv']['AUC'],
              'median' : performance['median']['AUC'],
              'gauss' : performance['gauss']['AUC']}
    
    colors = ['red', 'green', 'black', 'brown']
    i = 0
    
    
    for filter_type in ['standard', 'conv', 'median', 'gauss']:
        
        thr = thresholds[filter_type]
        x, y = [thr, thr], [0, h]
        
        plt.plot(x, y, marker='o', c=colors[i],
                 label='AUC: {:.3f} - Thr: {:.3f} - {}'.format(aucs[filter_type], thr, filter_type))
        
        i += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    