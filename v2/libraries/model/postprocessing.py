# -*- coding: utf-8 -*-

#%%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
import cv2
from copy import deepcopy
from time import time 
import pandas as pd

from scipy.stats import norm, gaussian_kde
from scipy.signal import medfilt
from scipy.ndimage import convolve, median_filter
from astropy.convolution import Gaussian2DKernel

import statsmodels.api as sm

from libraries.model.evaluate import evaluate
from libraries.model.evaluate import getThreshold, evaluateRoc
from libraries.model import evaluate as ev
from libraries.utils import ensure_folder, Paths, timeSpent
from libraries.model import score

paths = Paths()
filters = ['standard', 'conv', 'med', 'gauss']

empty_line = pd.DataFrame({'Filter':'',
                           'auc':'',
                           'prec':'',
                           'recall':'',
			               'iou':'',
                           'dice':''}, index=[0])
#%% FUNCTIONS

def res_table_init(save_folder):
    table = pd.DataFrame()
    table.to_excel(save_folder + 'Result_table.xlsx')

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

    
def computeFilters(as_map, params):

    as_maps = {}
    
    conv_map, med_map, gauss_map = {}, {}, {}
    
    std_as = np.array(list(as_map.values()))
    keys = list(as_map.keys())
    
    kernel = createKernel(params['conv'], dim=3)
    conv_as = convFilterScores(std_as, kernel)
    conv_map = createMap(keys, conv_as)
    
    kernel_size = params['med']
    med_as = medFilterScores(std_as, kernel_size)
    med_map = createMap(keys, med_as)
    
    sigma = params['gauss']
    gauss_as = gaussFilterScores(std_as, sigma)
    gauss_map = createMap(keys, gauss_as)
    
    as_maps['standard'] = as_map
    as_maps['conv'] = conv_map
    as_maps['med'] = med_map
    as_maps['gauss'] = gauss_map
    
    return as_maps

def saveFilter(sample, as_filtered, save_folder, figsize=(160,25),
               filter_name='filter'):
    
    start = time()
    
    filename = '{}_{}_anomaly score'.format(sample, filter_name.upper())
    folder = save_folder[str(sample)] + 'filters/'
    saveImage(as_filtered, folder, filename, figsize)
            
    end = time()

    timeSpent(end-start, 'Saving Filtered Anomaly Score')

def saveFilters(maps, save_folder, figsize=(160,25)):
    start = time()
    
    for f in filters:
        as_map = maps[f]
        
        for key in as_map:
            as_filtered = as_map[key]
            
            filename = '{}_{}_anomaly score'.format(key, f.upper())
            folder = save_folder[key] + 'filters/'
            saveImage(as_filtered, folder, filename, figsize)
            
    end = time()

    timeSpent(end-start, 'Saving Filtered Anomaly Score')   
    
def saveMask(mask, sample, save_folder):

    image = cv2.resize(mask, (1600,256), interpolation=cv2.INTER_LINEAR)
    filename = '{}_MASK_anomaly'.format(sample)
    saveImage(image, save_folder[str(sample)], filename)     
    
def saveMasks(masked_map, save_folder):

    for f in filters:
        for key in list(masked_map.keys()):
            image = masked_map[key]
            image = cv2.resize(image, (1600,256), interpolation=cv2.INTER_LINEAR)
            filename = '{}_{}_anomaly'.format(key, f.upper())
            saveImage(image, save_folder[key], filename) 


def saveAnomalyMap(anom_im, sample, save_folder, filter_name='filter'):
        
    image = cv2.resize(anom_im, (1600,256), interpolation=cv2.INTER_LINEAR)
    filename = '{}_{}_anomaly_detection'.format(sample, filter_name.upper())
    saveImage(image, save_folder[str(sample)], filename)

def saveAnomalyMaps(anom_maps, save_folder):
    
    for f in filters:
        anom_map = anom_maps[f]
        keys = list(anom_map.keys())
        
        for key in keys:
            image = anom_map[key]
            image = cv2.resize(image, (1600,256), interpolation=cv2.INTER_LINEAR)
            filename = '{}_{}_anomaly_detection'.format(key, f.upper())
            saveImage(image, save_folder, filename)
            
#            image = anom_map[key]
#            image = cv2.resize(image, (1600,256), interpolation=cv2.INTER_NEAREST)
#            filename = '{}_{}_anomaly_detection2'.format(key, f.upper())
#            saveImage(image, save_folder, filename)
#            
#            image = anom_map[key]
#            image = cv2.resize(image, (1600,256), interpolation=cv2.INTER_CUBIC)
#            filename = '{}_{}_anomaly_detection3'.format(key, f.upper())
#            saveImage(image, save_folder, filename)
#            
#            image = anom_map[key]
#            image = cv2.resize(image, (1600,256), interpolation=cv2.INTER_LANCZOS4)
#            filename = '{}_{}_anomaly_detection4'.format(key, f.upper())
#            saveImage(image, save_folder, filename)
#            
#            image = anom_map[key]
#            image = cv2.resize(image, (1600,256), interpolation=cv2.INTER_AREA)
#            filename = '{}_{}_anomaly_detection5'.format(key, f.upper())
#            saveImage(image, save_folder, filename)

def saveFullMask(full_mask, sample, save_folder, filter_name='filter'):

    filename = '{}_{}_full_mask'.format(sample, filter_name.upper())
    saveImage(full_mask, save_folder[str(sample)], filename)
    saveImage(full_mask, save_folder[str(sample)] + '/comparison/', filename)
    
def saveFullMasked(full_masked_maps, save_folder):
    
    for f in filters:
        anom_map = full_masked_maps[f]
        keys = list(anom_map.keys())
        
        for key in keys:
            filename = '{}_{}_full_masked_map'.format(key, f.upper())
            saveImage(anom_map[key], save_folder, filename)
            saveImage(anom_map[key], save_folder + '/comparison/', filename)
        
def saveImage(image, save_folder, filename, figsize=(160,25)):
    fig = plt.figure(frameon=False, figsize=figsize)
    plt.axis('off')
    plt.imshow(image)
    print('> Saving \'{}\' at {}'.format(filename, save_folder))
    plt.savefig(save_folder + filename + '.png',
                bbox_inches='tight', transparent=True,
                pad_inches=0, dpi=10)
    plt.close(fig)
    plt.show()    

def createMap(keys, anom_scores):
    
    as_map = {}
    
    for i in range(len(keys)):
        as_map[keys[i]] = anom_scores[i]
        
    return as_map

def hist_data(data, bins, my_range, thr=None, title='', color='r', label='',
              figsize=(12,3), plot=True, save_folder=None, filename=None):
    
    fig = plt.figure(figsize=figsize)
    plt.title('Histogram ' + title)
    hist = plt.hist(data, bins=bins, color='#ffcc99', 
                    range=my_range, density=True, label='Anomaly Score')
    
    if(isinstance(thr, list) and isinstance(color, list) and isinstance(label, list)):
        for i in range(len(thr)):
            lab = label[i] + ': {:.4f}'.format(thr[i])
            
            plt.plot([thr[i], thr[i]], [0, max(hist[0])], c=color[i],
                     marker='o', label=lab)
    
    elif(thr is not None):
        label = label + ': {:.4f}'.format(thr)
        plt.plot([thr, thr], [0, max(hist[0])], c=color, marker='o', label=label)
    
    plt.legend(loc='best')
    if(save_folder is not None and filename is not None):
        print('> Saving \'{}\' at {}'.format(filename, save_folder + 'histogram'))
        plt.savefig(save_folder + 'histogram/' + filename + '.png')

    if(plot==False):
        plt.close(fig)
    
    plt.show()
    
    return hist


def computeThreshold(as_filtered, kernel, hist_params, prob, save_folder=None):
    
    # COMPUTE FILTERED ANOMALY SCORES
    
    # HISTOGRAM FOR EACH MAP
    hist = hist_data(as_filtered.ravel(), hist_params['bins'], hist_params['range'],
                     title='Filtered Score', figsize=hist_params['figsize'], plot=False)
    
    # THRESHOLDS FOR EACH MAP
    thr = getThreshold(as_filtered.ravel(), prob, hist)
    
    # HISTOGRAM WITH THRESHOLDS
    hist_data(as_filtered.ravel(), hist_params['bins'], hist_params['range'], thr=thr,
              label='Anomaly Score',
                 title='Filtered Score', color='r', figsize=hist_params['figsize'], plot=True,
                 save_folder=save_folder, filename='{}_histogram'.format('Filtered Score'))
    
    return thr


def computeThresholds(as_map, kernel_params, hist_params, prob, save_folder=None):
    
    maps = {}
    thrs = {}
    
    # COMPUTE FILTERED ANOMALY SCORES
    maps = computeFilters(as_map, kernel_params)

    
    labels = ['Standard', 'Conv       ', 'Median   ', 'Gaussian']
    colors = ['brown', 'r', 'b', 'green']
    
    i = 0
    for f in filters:
        as_array = np.array(list(maps[f].values()))
        
        # HISTOGRAM FOR EACH MAP
        hist = hist_data(as_array.ravel(), hist_params['bins'], hist_params['range'],
                         title=f.upper(), figsize=hist_params['figsize'], plot=False)
        
        # THRESHOLDS FOR EACH MAP
        thr = getThreshold(as_array.ravel(), prob, hist)
        thrs[f] = thr
        
        # HISTOGRAM WITH THRESHOLDS
        hist_data(as_array.ravel(), hist_params['bins'], hist_params['range'], thr=thr, label=f,
                     title=f.upper(), color=colors[i], figsize=hist_params['figsize'], plot=True,
                     save_folder=save_folder, filename='{}_histogram'.format(f.upper()))
        
        i += 1
    
    # ALL THRESHOLDS
    std_as = np.array(list(as_map.values()))
    hist_data(std_as.ravel(), hist_params['bins'], hist_params['range'],
                     thr=list(thrs.values()),label=labels, color=colors, title='Anomaly Scores',
                     figsize=hist_params['figsize'], plot=True, save_folder=save_folder,
                     filename='Anomaly_score_thrs')
    
    return thrs

def tuning_conv_filter(as_map, gt_map):
    
    ks = np.arange(3, 23, 2)
    best = {'auc':0,
            'k':0}
    
    for k in ks:
        kernel = createKernel(k, dim=3)
        conv_map = convFilterScores(as_map, kernel)
        
        auc, _ = evaluateRoc(conv_map.ravel(), gt_map.ravel(), plot=False)
    
        if(auc > best['auc']):
            best['auc'] = auc
            best['k'] = k
            
    return best

def tuning_med_filter(as_map, gt_map):
    
    ks = np.arange(3, 23, 2)
    best = {'auc':0,
            'k':0}
    
    for k in ks:
        conv_map = medFilterScores(as_map, k)
        
        auc, _ = evaluateRoc(conv_map.ravel(), gt_map.ravel(), plot=False)
    
        if(auc > best['auc']):
            best['auc'] = auc
            best['k'] = k
            
    return best

def tuning_gauss_filter(as_map, gt_map):
    
    sigmas = np.arange(1, 10, 0.1)
    best = {'auc':0,
            'sigma':0}
    
    for sigma in sigmas:
        conv_map = gaussFilterScores(as_map, sigma)
        
        auc, _ = evaluateRoc(conv_map.ravel(), gt_map.ravel(), plot=False)
    
        if(auc > best['auc']):
            best['auc'] = auc
            best['sigma'] = sigma
            
    return best

def compute_filter_anomalies(a_score, mask, thr, info='', save_folder=None):
    
    anom_image = a_score > thr
    anom_image = anom_image * 1
    anom_image = anom_image.astype(np.float32)
    
    auc = evaluateRoc(a_score.ravel(), mask.ravel(),
                      info=info, thr=thr, save_folder=save_folder)
    
    precision = ev.precision(mask.ravel(), anom_image.ravel())
    iou = ev.IoU(mask, anom_image)
    recall = ev.recall(mask.ravel(), anom_image.ravel())
    dice = ev.dice(mask, anom_image)
    
    result = {'auc':auc[0], 
              'prec':precision,
	          'recall':recall,
              'iou':iou,
              'dice':dice}
    
    return anom_image, result

def compute_anomalies(as_map, gt_map, index, thr, info='', save_folder=None):
    
    print('> {} Anomaly Scores'.format(info.upper()))
    
    keys = list(as_map.keys())
    key = keys[index]
    as_array = np.array(list(as_map[key]))
    gt_array = np.array(list(gt_map[key]))
    
    anom_image = as_array > thr
    anom_image = anom_image * 1
    anom_image = anom_image.astype(np.float32)
    
    auc = evaluateRoc(as_array.ravel(), gt_array.ravel(),
                      info=info, thr=thr, save_folder=save_folder)
    
    precision = ev.precision(gt_array.ravel(), anom_image.ravel())
    iou = ev.IoU(gt_array, anom_image)
    recall = ev.recall(gt_array.ravel(), anom_image.ravel())
    dice = ev.dice(gt_array, anom_image)
    
    result = {'auc':auc[0], 
              'prec':precision,
	      'recall':recall,
              'iou':iou,
              'dice':dice}
    
    anom_image_dict = {key:anom_image}
    
    return anom_image_dict, result
    
def compute_anomalies_all_filters(index, gt_map, as_maps, thrs, save_folder=None):
    '''
        Computing results per filters
    '''
    anomaly_map = {}
    results = {}
    
    for f in filters:
        as_map = as_maps[f]        
        thr = thrs[f]
        
        anomaly_map[f], results[f] = compute_anomalies(as_map, gt_map, index,
                   thr, info=f.upper(), save_folder=save_folder)
        
    return anomaly_map, results

def resultsPerEvaluation_single(results):
    
    '''
        Computing results per evaluation('auc', 'prec', ...)
    '''
    
    
    auc = {}
    prec = {}
    recall = {}
    iou = {}
    dice = {}
    
    readable_results = {}

    auc = results['auc']
    prec = results['prec']
    recall = results['recall']
    iou = results['iou']
    dice = results['dice']
        
    readable_results  = {'auc':auc,
                           'prec':prec,
                           'recall':recall,
                           'iou':iou,
                           'dice':dice}
    
    return readable_results 

def resultsPerEvaluation(results):
    
    '''
        Computing results per evaluation('auc', 'prec', ...)
    '''
    
    filters = ['standard', 'conv', 'med', 'gauss']
    
    auc = {}
    prec = {}
    recall = {}
    iou = {}
    dice = {}
    
    readable_results = {}
    
    for f in filters:
        auc[f] = results[f]['auc']
        prec[f] = results[f]['prec']
        recall[f] = results[f]['recall']
        iou[f] = results[f]['iou']
        dice[f] = results[f]['dice']
        
    readable_results  = {'auc':auc,
                           'prec':prec,
                           'recall':recall,
                           'iou':iou,
                           'dice':dice}
    
    return readable_results 

def readEvaluation(criterium, eval_results):
    
    assert criterium in ['auc', 'prec', 'recall', 'iou', 'dice'], 'Evaluation has to be one of these: auc, prec, recall, iou, dice'
    
    filters = ['standard', 'conv', 'med', 'gauss'] 
    best = {'filter':'',
            'value':0}
    
    print('**************************')
    print('> -------- {} --------'.format(criterium.upper()))
    print('>')
    
    for f in filters:
        perf = eval_results[criterium][f]
        if(perf > best['value']):
            best['value'] = perf
            best['filter'] = f.upper()
            
        print('> {}   \t: {:.3f}'.format(f.upper(), eval_results[criterium][f]))
        
    print('>')
    print('> Best filter: {}'.format(best['filter']))
    print('**************************')
    
    return best
    
def best_performance(evaluation):
    
    performance = ['auc', 'prec', 'recall', 'iou', 'dice']
    
    bests = {}
    
    for perf in performance:
        bests[perf] = readEvaluation(perf, evaluation)
    
    return bests
    
def overlapAnomalies_single(anom_im, mask, image):
    h, w, _ = image.shape
    
    resized_anom = cv2.resize(anom_im, (w,h),
                              interpolation=cv2.INTER_LINEAR)

    resized_mask = cv2.resize(mask, (w,h),
                              interpolation=cv2.INTER_LINEAR)

#    plt.imshow(resized_anom)
#    plt.show()
#
#    plt.imshow(resized_mask)
#    plt.show()

    masked_im = deepcopy(image)
    masked_im[resized_anom==1, 0] = 255
    masked_im[resized_mask==1, 2] = 255

#    plt.imshow(masked_im)
#    plt.show()
    
    return masked_im

def overlapAnomalies(index, masked_maps, anomaly_maps, interp=cv2.INTER_LINEAR):
    keys = list(masked_maps.keys())
    key = keys[index]
    
    h, w, _ = masked_maps[key].shape
    
    full_masked_maps = {}
    
    for f in filters: 
#        plt.imshow(anomaly_map[f])
#        plt.show()
        
        resized_anom = cv2.resize(anomaly_maps[f][key], (w,h), interpolation=interp)
#        plt.imshow(resized_anom)
#        plt.show()
        
#        resized_anom = cv2.resize(anomaly_maps[f][key], (w,h), interpolation=cv2.INTER_NEAREST)
#        plt.imshow(resized_anom)
#        plt.show()
        
#        resized_anom = cv2.resize(anomaly_map[f], (w,h), interpolation=cv2.INTER_CUBIC)
#        plt.imshow(resized_anom)
#        plt.show()
#        
#        resized_anom = cv2.resize(anomaly_map[f], (w,h), interpolation=cv2.INTER_AREA)
#        plt.imshow(resized_anom)
#        plt.show()
#        
#        resized_anom = cv2.resize(anomaly_map[f], (w,h), interpolation=cv2.INTER_LANCZOS4)
#        plt.imshow(resized_anom)
#        plt.show()
        
        
#        plt.imshow(resized_anom)
#        plt.show()
        masked_im = deepcopy(masked_maps[key])
        masked_im[resized_anom==1, 0] = 255
#        plt.imshow(masked_im)
#        plt.show()
        
        masked_map = {key:masked_im}
        full_masked_maps[f] = masked_map
        
#        plt.imshow(masked_images[f])
#        plt.show()
        
    return full_masked_maps

def complete_evaluation(index, gt_map, as_maps, masked_map, thrs, key, save_folder=None):
    save = save_folder is not None
    
    anomaly_maps, res_per_filter = compute_anomalies_all_filters(index, gt_map, as_maps, thrs, save_folder[key])
    
    full_masked_maps = overlapAnomalies(index, masked_map, anomaly_maps)
    
    
    if(save):
        saveAnomalyMaps(anomaly_maps, save_folder[key])
#        saveMasks(masked_map, save_folder)
        saveFullMasked(full_masked_maps, save_folder[key])
       
    # RESULTS
    evaluation = resultsPerEvaluation(res_per_filter)
    bests = best_performance(evaluation)
    
    # TABLE RESULTS
    fillResultTable(res_per_filter, bests, key, save_folder)
    
    return anomaly_maps, full_masked_maps, res_per_filter, evaluation, bests

def fillResultTable_single(res, n_image, save_folder):
    filepath = save_folder['general'] + 'Result_table.xlsx'
    table = pd.read_excel(filepath, index_col=0)
    
    res_df = pd.DataFrame([res])
    res_df = res_df.round(3)   
       
    res_df.insert(0, 'Image', n_image)
    res_df = res_df.set_index('Image')
    
    table = pd.concat([table, res_df]) 
    table.to_excel(filepath)


def fillResultTable(res, bests, key, save_folder):
    filepath = save_folder['general'] + 'Result_table.xlsx'
    table = pd.read_excel(filepath, index_col=0)
    
    res_df = pd.DataFrame(res)
    res_df = res_df.round(3)

    bests_df = pd.DataFrame(bests)
    bests_df = bests_df.transpose()    
        
    df = res_df.merge(bests_df, left_index=True, right_index=True)   
    df = df.transpose()
    
    df.insert(0, 'Filter', ['Standard', 'Conv', 'Med', 'Gauss', 'Best_Filter', 'Value'])    
    df.insert(0, 'Image', key)
    df = df.set_index('Image')
    
    table = pd.concat([table, empty_line, df]) 
    table.to_excel(filepath)
  

def plotAnomalies(as_filters, anomaly_map, masked_image, index, figsize=(8,15), bests=None):
    
    filters = ['standard', 'conv', 'med', 'gauss'] 

    for f in filters:
        plt.figure(figsize=figsize)
        plt.title('{} Anomaly Scores'.format(f.upper()))
        plt.imshow(as_filters[f][index])
        plt.show()
        
        plt.figure(figsize=figsize)
        plt.title('{} Detection'.format(f.upper()))
        plt.imshow(anomaly_map[f])
        plt.show()
        
        plt.figure(figsize=figsize)
        plt.title('{} Detection'.format(f.upper()))
        plt.imshow(masked_image[f])
        plt.show()
        
        
    if(bests):
        display(bests)
    
def saveAnomalies(as_filters, anomaly_map, masked_image, index,
                  folder_save, figsize=(18,3)):
    
    filters = ['standard', 'conv', 'med', 'gauss'] 

    for f in filters:
        # ANOMALY SCORE
        fig = plt.figure(figsize=figsize)
        plt.title('{} Anomaly Scores'.format(f.upper()))
        plt.axis('off')
        plt.imshow(as_filters[f][index])
        
        filename = '{}_Anomaly Scores.png'
        plt.savefig(folder_save + filename, transparent=True, dpi=1000)
        
        plt.close(fig)
        plt.show()
        
        # ANOMALY DETECTION
        fig = plt.figure(figsize=figsize)
        plt.title('{} Detection'.format(f.upper()))
        plt.axis('off')
        plt.imshow(anomaly_map[f])

        filename = '{}_Anomaly Detector.png'
        plt.savefig(folder_save + filename, transparent=True, dpi=1000)
                
        plt.close(fig)
        plt.show()
        
        # FULL MASKED IMAGE
        fig = plt.figure(figsize=figsize)
        plt.title('{} Detection'.format(f.upper()))
        plt.axis('off')
        plt.imshow(masked_image[f])
        
        filename = '{}_Anomaly Detection.png'
        plt.savefig(folder_save + filename, transparent=True, dpi=1000)
        
        plt.close(fig)
        plt.show()
    
def setSaveFoldersResults(samples, folder_extra_name=None):
    
    save_folders = {}
    
    for sample in samples:
        save_folder = '{}{}_images/{}/'.format(paths.results_path, str(len(samples)), str(sample))
        
        if(folder_extra_name is not None):
            save_folder = '{}{}/{}_images/{}/'.format(paths.results_path, folder_extra_name,
                                                       str(len(samples)), str(sample))
        
        ensure_folder(save_folder)
        ensure_folder(save_folder + 'filters/')
        ensure_folder(save_folder + 'comparison/')
        save_folders[str(sample)] = save_folder
        
    save_folders['general'] = paths.results_path + '{}/{}_images/'.format(folder_extra_name,
                                                                str(len(samples)))
    ensure_folder(save_folders['general'])
    ensure_folder(save_folders['general'] + 'histogram/')
    
    return save_folders
    
def writeResults(res, ev,  bests, key, save_folder):
    
    filename = 'data_results: {}.txt'.format(key)
    
    content = '\t\tImage id: {}\n\n\n'.format(key)
    
    for f in filters:
        result = res[f]
        content = content + '-------- {} --------\n\n'.format(f.upper())
        content = content + '- AUC:     {:.3f}\n'.format(result['auc'])
        content = content + '- Prec:    {:.3f}\n'.format(result['prec'])
        content = content + '- Recall:  {:.3f}\n'.format(result['recall'])
        content = content + '- Iou:     {:.3f}\n'.format(result['iou'])
        content = content + '- Dice:    {:.3f}\n\n'.format(result['dice'])
#        content = content + '-  -  -  -  -  -  -  -  -\n\n'
        
    content = content + '_________________________\n\n\n'
        
    content = content + '--- Best Performance ---\n\n'
    
    for perf in bests:
        content = content + '- {}:  \t{:.3f}   -->   {}\n'.format(perf, bests[perf]['value'], bests[perf]['filter'])
    
    content = content + '\n\n'
    content = content + '_______________________\n\n\n'
    
    content = content + '--- Ordered by Performance ---\n\n'
    
    for perf in ev:
        content = content + '------- {} -------\n\n'.format(perf.upper())
        content = content + '- Standard:  {:.3f}\n'.format(ev[perf]['standard'])
        content = content + '- Conv:      {:.3f}\n'.format(ev[perf]['conv'])
        content = content + '- Median:    {:.3f}\n'.format(ev[perf]['med'])
        content = content + '- Gaussian:  {:.3f}\n\n'.format(ev[perf]['gauss'])
    
    print('Savind data at {}'.format(save_folder))
    f = open(save_folder + filename, 'a')
    f.write(content)
    f.close()
    
    f = open(save_folder + 'comparison/' + filename, 'a')
    f.write(content)
    f.close()
    
#%% SCORE

def anomalyDetection(model, image, mask, stride, patch_size, kernel, filtering='conv'):
    
    as_score, mask = score.anomalyScoreFromImage(model, image, mask, stride, patch_size)
    
    if(filtering=='conv'):
        as_filter = convFilterScores(as_score, kernel)
        
    anom_det = as_filter > model.threshold
    anom_det = anom_det * 1
    anom_det = anom_det.astype(np.float32)
    
    return as_score, mask, anom_det
    
#def detector(as_filter, threshold):
#    
##    try:
##        as_filter = as_filter.cpu().detach().numpy()
###        print(as_filter)
##    except:
##        print('-asasasa---a-sasas--as-')
#    
#    
##    as_filter = as_filter.cpu().detach().numpy()
#    torch.where(as_filter > threshold, 1, 0)
#    as_filter[as_filter > threshold]=1
#    as_filter[as_filter <= threshold]=0
#    
##    anom_det = anom_det * 1
##    anom_det = anom_det.astype(np.float32)
##    anom_det = anom_det
#    
#    return as_filter
    
    
#%%
def loadDefectImages():
    path = '../'
    selected_data = pd.read_excel(path + 'defect.xlsx', index_col=0)    
        
    n_images = selected_data['N° image'].to_list()
    defects = selected_data.index.to_list()
    
    images = {'Defect':defects,
              'n_images':n_images}
    
    return images
    
    
