# -*- coding: utf-8 -*-

#%%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

from scipy.stats import norm, gaussian_kde
from scipy.signal import medfilt
from scipy.ndimage import convolve, median_filter
from astropy.convolution import Gaussian2DKernel

import statsmodels.api as sm

from libraries.model.evaluate import evaluate
from libraries.model.evaluate import getThreshold
from libraries.utils import ensure_folder
#%% FUNCTIONS

def createKernel(kernel_size, dim=2):
    
    assert dim == 2 or dim == 3, 'Wrong dimensione param'
    
    if(dim==2):
        kernel = np.ones([kernel_size, kernel_size])
        
    elif(dim==3):
        kernel = np.ones([1, kernel_size, kernel_size])
    
    return kernel

def convFilterScores(scores, kernel):
    
#    conv_scores = convolve(scores, kernel, mode='constant')
#    conv_scores = convolve(scores, kernel, mode='nearest')
    conv_scores = convolve(scores, kernel)
        
    return conv_scores


def medFilterScores(scores, kernel_size):
    
    size = (1, kernel_size, kernel_size)
    
    med_scores = median_filter(scores, size)

    return med_scores

def gaussFilterScores(scores, sigma):
    
    kernel = Gaussian2DKernel(sigma).array
    kernel = kernel.reshape((1, kernel.shape[0], kernel.shape[1]))
    
    gauss_scores = convolve(scores, kernel)

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
               folder_save=None, bins=1000, h=None, x_limit=None):
    
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
    
    # HISTOGRAM
    values, _, _ = plt.hist([anomalies, normals], bins=bins,
             label=['Anomaly Scores', 'Normal Scores'], density=True)
    
    if(x_limit is None):
        x_limit = np.mean(anomalies) + np.mean(anomalies)/2
    
    if(h is None):
        h = max(values[1])
    
    __plottingThresholds(performance, h)
    
    plt.xlim(0, x_limit)
    plt.ylim(0, h)
    plt.legend(loc='best')
    plt.xlabel('Score')
    plt.show()
    
    
    # DISTRIBUTIONS     
    plt.figure(figsize=figsize)
    
    n_mean, n_std = norm.fit(normals)
    a_mean, a_std = norm.fit(anomalies)
    
    x = np.arange(0, x_limit, x_limit/bins)
    
    norm_distr = norm(n_mean, n_std)
    anom_distr = norm(a_mean, a_std)
    
    __plottingDistributions(x, norm_distr, anom_distr)
   
    # THRESHOLD
    __plottingThresholds(performance, h=max(norm_distr.pdf(x)), distr=norm_distr)
    plt.legend()
    
    if(folder_save is not None):
        ensure_folder(folder_save)
        print('> Saving Distribution Score at .. {}'.format(folder_save))
        plt.savefig(folder_save + 'distribution.png')
    plt.show()


    plt.figure(figsize=figsize)
    
    __plottingThresholds(performance, h)
    
    sn.distplot(anomalies, bins=bins, kde=False,
                hist=True, norm_hist=True, label='Anomaly Score')
    sn.distplot(normals, bins=bins, kde=False,
                hist=True, norm_hist=True,label='Normal Score')

    plt.xlim(0, x_limit)
    plt.ylim(0, h)
    plt.legend()
    
    if(folder_save is not None):
        ensure_folder(folder_save)
        print('> Saving Distribution Score at .. {}'.format(folder_save))
        plt.savefig(folder_save + 'histogram.png')
    
    plt.xlabel('Score')
    plt.show()
    
    return 
    
def __plottingDistributions(x, norm_distr, anom_distr, c_norm='#ffcc99', c_anom='#dceaf9'):
    
    pdf_norm = norm_distr.pdf(x)
    pdf_anom = anom_distr.pdf(x)
    
    plt.fill_between(x, pdf_anom, color=c_anom, label='Normal Distr')
    plt.plot(x, pdf_anom, c=c_anom, ls='--')
    
    plt.fill_between(x, pdf_norm, color=c_norm, label='Anomaly Distr')
    plt.plot(x, pdf_norm, c=c_norm)

def __plottingThresholds(performance, h, distr=None):

    
    thresholds = {'standard' : performance['standard']['Threshold'],
                  'conv' : performance['conv']['Threshold'],
                  'median' : performance['median']['Threshold'],
                  'gauss' : performance['gauss']['Threshold']}
    
    aucs = {'standard' : performance['standard']['AUC'],
              'conv' : performance['conv']['AUC'],
              'median' : performance['median']['AUC'],
              'gauss' : performance['gauss']['AUC']}
    
    colors = ['r', 'green', 'black', 'brown']
    i = 0
    
    
    for filter_type in ['standard', 'conv', 'median', 'gauss']:
        
        thr = thresholds[filter_type]
        x, y = [thr, thr], [0, h]
        
        if(distr is None):
            label = 'AUC: {:.3f} - Thr: {:.3f} - {}'.format(aucs[filter_type], thr, filter_type)
        else:
            cdf = distr.cdf(thr)
            label = 'CDF: {:.2f}% - AUC: {:.3f} - Thr: {:.3f} - {}'.format(cdf, aucs[filter_type], thr, filter_type)
            
        plt.plot(x, y, marker='o', c=colors[i], label=label)
        
        i += 1

def computeFilters(as_map, params):
    
    kernel = createKernel(params['conv'], dim=3)
    conv_map = convFilterScores(as_map, kernel)
    
    kernel_size = params['med']
    med_map = medFilterScores(as_map, kernel_size)
    
    sigma = params['gauss']
    gauss_map = gaussFilterScores(as_map, sigma)
    
    return conv_map, med_map, gauss_map

def hist_data(data, bins, my_range, thr=None, title='', color='r', label=''):
    
    plt.title('Histogram ' + title)
    hist = plt.hist(data, bins=bins, color='#ffcc99', 
                    range=my_range, density=True)
    
    if(isinstance(thr, list) and isinstance(color, list) and isinstance(label, list)):
        for i in range(len(thr)):
            lab = label[i] + ': {:.3f}'.format(thr[i])
            
            plt.plot([thr[i], thr[i]], [0, max(hist[0])], c=color[i],
                     marker='o', label=lab)
    
    elif(thr is not None):
        label = label + ': {:.3f}'.format(thr)
        plt.plot([thr, thr], [0, max(hist[0])], c=color, marker='o', label=label)
    
    plt.legend(loc='best')
    
    plt.show()
    
    return hist

def computeThresholds(as_map, kernel_params, hist_params, prob):
    
    conv_map, med_map, gauss_map = computeFilters(as_map, kernel_params)
    
    std_hist = hist_data(as_map.ravel(), hist_params['bins'], hist_params['range'], title='Standard')
    conv_hist = hist_data(conv_map.ravel(), hist_params['bins'], hist_params['range'], title='Conv')
    med_hist = hist_data(med_map.ravel(), hist_params['bins'], hist_params['range'], title='Median')
    gauss_hist = hist_data(gauss_map.ravel(), hist_params['bins'], hist_params['range'], title='Gaussian')
    
    std_thr = getThreshold(as_map.ravel(), prob, std_hist)
    conv_thr = getThreshold(conv_map.ravel(), prob, conv_hist)
    med_thr = getThreshold(med_map.ravel(), prob, med_hist)
    gauss_thr = getThreshold(gauss_map.ravel(), prob, gauss_hist)
    
    thrs = [std_thr, conv_thr, med_thr, gauss_thr]
    labels = ['Standard', 'Conv       ', 'Median   ', 'Gaussian']
    colors = ['brown', 'r', 'b', 'green']
    
    # STANDARD THRESHOLD
    hist_data(as_map.ravel(), hist_params['bins'], hist_params['range'], thr=std_thr,label='Standard',
                     title='Conv', color=colors[0])
    # CONV THRESHOLD
    hist_data(conv_map.ravel(), hist_params['bins'], hist_params['range'], thr=conv_thr,label='Conv',
                     title='Conv', color=colors[1])
    # MEDIAN THRESHOLD
    hist_data(med_map.ravel(), hist_params['bins'], hist_params['range'], thr=med_thr,label='Med',
                     title='Med', color=colors[2])
    # GAUSSIAN THRESHOLD    
    hist_data(gauss_map.ravel(), hist_params['bins'], hist_params['range'], thr=gauss_thr,label='Gauss',
                     title='Gauss', color=colors[3])
    
    # ALL THRESHOLDS
    hist_data(as_map.ravel(), hist_params['bins'], hist_params['range'],
                     thr=thrs,label=labels, color=colors, title='Anomaly Scores')
    
    return std_thr, conv_thr, med_thr, gauss_thr

    
    
    
    
    
    
    
    
    
    