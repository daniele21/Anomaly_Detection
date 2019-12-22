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
from libraries.dataset_package import dataset_manager as dm
from libraries.dataset_package.dataset_manager import Shape
from libraries.model.dataset import generateDataloaderTest

paths = Paths()
csv_directory = paths.csv_directory
#%%
def computeEvaluation(mask_true, mask_pred, info, folder_save):
    
    mask_true.dtype=int
    mask_pred.dtype=int
    
    prec = precision(mask_true.ravel(), mask_pred.ravel())
    acc = accuracy(mask_true.ravel(), mask_pred.ravel())
    rec = recall(mask_true.ravel(), mask_pred.ravel())
    iou = IoU(mask_pred, mask_true)
    
    results = {'acc':acc,
               'prec':prec,
               'rec':rec,
               'iou':iou,
               'info': info}
    
    writeDataResults(results, folder_save)

def evaluateResult(model, img, mask):
  
    # 2.PREDICTION ANOMALY THRESHOLDING OVER EACH BATCH
#    dataTest = generateDataloaderTest(img.patches, model.opt)
#    pred_patches, _ , performance = model.predictImage(dataTest, img.folder_save)


    # 1.PREDICTION ANOMALY USING THE FINAL THRESHOLD OF MODEL 
    # OVER ALL DATA TRAINED
    outcomes = []
    thr_dict = {}
    
    for x in ['standard','conv','median']:
        thr_dict.update({x : model.performance[x]['Threshold']})
    
    for x in thr_dict:
        thr = thr_dict[x]
        
        # ANOMALY PREDICTION FOR EACH SINGLE PATCH
        for patch in tqdm(img.patches, total=len(img.patches)):
        
            patch_image = patch.patch_image
            outcome, score, threshold = model.predict(patch_image, threshold=thr)
            outcomes.append(outcome)
            patch.anomaly = outcome[1]
            patch.setScore(score, threshold)
            
        # DRAWING ANOMALIES --> METHOD 1
        
        # SIMPLE METHOD
        simple_mask_1, n_anom_1 = img.drawAnomaliesSimple()
        
#        return simple_mask_1, mask
        
        info = 'Thr over data - ' + x
        computeEvaluation(mask, simple_mask_1, info, img.folder_save)
        
        if(x == 'standard'):
            # MAJORITY VOTING
            maj_mask_1 = img.drawAnomaliesMajVoting()
            
            # EVALUATION
            info = 'Thr over data - MAJORITY VOTING'
            computeEvaluation(mask, maj_mask_1, info, img.folder_save)

    
def automaticEvaluation(model, start, end, stride):
    
    shape = Shape(32,32)
    model_name = model.opt.name
    
    train = pd.read_csv(csv_directory + 'train_unique.csv', index_col=0)
    
    for index in range(start, end):        
        img, _, mask = dm.extractPatchesForTest(train, index, shape, stride, model_name)
        evaluateResult(model, img, mask)
        

