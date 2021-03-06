#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:23:30 2019

@author: daniele
"""
#%%
import os
import pandas as pd
from tqdm import tqdm

from libraries.model.ganomaly import loadModel, outputSample
from libraries.utils import Paths, writeDataResults
from libraries.model.evaluate import evaluate, accuracy, precision, recall, IoU
from libraries.model.evaluate import precision_recall
from libraries.dataset_package import dataset_manager as dm
from libraries.dataset_package.dataset_manager import Shape
from libraries.model.dataset import generateDataloaderTest

paths = Paths()
csv_directory = paths.csv_directory
#%%
def computeEvaluation(mask_true, mask_pred, info, stride, folder_save):
    
    mask_true = mask_true.astype(int)
    mask_pred = mask_pred.astype(int)
    
    avg_prec = precision_recall(mask_true.ravel(), mask_pred.ravel(),
                                plot=True, folder_save=folder_save)
    
    prec = precision(mask_true.ravel(), mask_pred.ravel())
    acc = accuracy(mask_true.ravel(), mask_pred.ravel())
    rec = recall(mask_true.ravel(), mask_pred.ravel())
    iou = IoU(mask_pred, mask_true)
    
    results = {'acc':acc,
               'prec':prec,
               'rec':rec,
               'iou':iou,
               'avg_prec':avg_prec,
               'info': info}
    
    writeDataResults(results, stride, folder_save)

def evaluateResult(model, img, mask, stride, folder_save):
  
    # 2.PREDICTION ANOMALY THRESHOLDING OVER EACH BATCH
#    dataTest = generateDataloaderTest(img.patches, model.opt)
#    pred_patches, _ , performance = model.predictImage(dataTest, img.folder_save)


    # 1.PREDICTION ANOMALY USING THE FINAL THRESHOLD OF MODEL 
    # OVER ALL DATA TRAINED
    outcomes = []
    thr_dict = {}
    
    for x in ['standard','conv','median','gauss']:
        
        try:
            thr_dict.update({x : model.performance[x]['Threshold']})
            
        except:
            break
    
    for x in thr_dict:
        print('> {} Filter Evaluation'.format(x))
        try:
            thr = thr_dict[x]
            
        except:
            break

        # ANOMALY PREDICTION FOR EACH SINGLE PATCH
        for patch in img.patches:
        
            patch_image = patch.patch_image
            outcome, score, threshold = model.predict(patch_image, threshold=thr)
            outcomes.append(outcome)
            patch.anomaly = outcome[1]
            patch.setScore(score, threshold)
            
        # DRAWING ANOMALIES --> METHOD 1
        print('threshold: {}'.format(threshold))
        if(x == 'conv' or x == 'median' or x == 'gauss'):
            try:
                x = str(x) + '_param=' + str(model.performance[x]['param'])
            except:
                x = str(x) + '_param=' + str(model.performance[x]['k'])
            
        # SIMPLE METHOD
        simple_mask_1, n_anom_1 = img.drawAnomaliesSimple(info = x)
        
#        return simple_mask_1, mask
        
        info = '\tStride:{} - {}\n'.format(stride, x)
        computeEvaluation(mask, simple_mask_1, info, stride, folder_save)
        
#        if(x == 'standard'):
#            # MAJORITY VOTING
#            maj_mask_1 = img.drawAnomaliesMajVoting()
#            
#            # EVALUATION
#            info = 'Thr over data - MAJORITY VOTING'
#            computeEvaluation(mask, maj_mask_1, info, img.folder_save)

    
def automaticEvaluation(model, samples, stride, folder_save=None):
    
    if(folder_save is None):
        folder_save = '../../{}-results/'.format(model.opt.name)
    else:
        folder_save = folder_save
    
    shape = Shape(32,32)
    model_name = model.opt.name
    
    train = pd.read_csv(csv_directory + 'train_unique.csv', index_col=0)
    
    for index in samples:        
        img, _, mask = dm.extractPatchesForTest(train, index, shape, stride,
                                                model_name, folder_save)
        
        evaluateResult(model, img, mask, stride, img.folder_save)
        

