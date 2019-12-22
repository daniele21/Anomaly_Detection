#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:33:05 2019

@author: daniele
"""

#%% IMPORTS
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
        
        
    
#%% LOAD MODEL
filename = 'Ganom_v12.0_lr:1e-06|Epoch:205|Auc:0.901|Loss:176.7352.pth.tar'
adModel = loadModel(filename)

#%%
automaticEvaluation(adModel, 1070, 1075, 4)
automaticEvaluation(adModel, 1075, 1080, 4)
automaticEvaluation(adModel, 1080, 1085, 4)
automaticEvaluation(adModel, 1085, 1090, 4)

#%% GET PATCHES FROM AN IMAGE

train = pd.read_csv(csv_directory + 'train_unique.csv', index_col=0)
index = 1060
shape = Shape(32,32)
stride = 32
model_name = adModel.opt.name

img, _, mask = dm.extractPatchesForTest(train, index, shape, stride, model_name)
#%%
evaluateResult(adModel, img, mask)
#%%
dataTest = generateDataloaderTest(img.patches, adModel.opt)

pred_patches, samples, performance = adModel.predictImage(dataTest, img.folder_save)

simple_mask, n_anom = img.drawAnomaliesSimple()
img.drawAnomaliesSimple(pred_patches, info='TH-PER-BATCH')

maj_vot_mask = img.drawAnomaliesMajVoting()
maj_vot_mask2 = img.drawAnomaliesMajVoting(pred_patches, info='TH-PER-BATCH')

evaluate(mask.ravel(), maj_vot_mask2.ravel(),
         plot=True, folder_save=img.folder_save)

evaluate(mask, maj_vot_mask, metric='precision')
evaluate(mask, maj_vot_mask2, metric='precision')
evaluate(mask.ravel(), maj_vot_mask2.ravel(), metric='precision')
evaluate(mask.ravel(), maj_vot_mask2.ravel(), metric='acc')

precision(mask.ravel(), maj_vot_mask2.ravel())
accuracy(mask.ravel(), maj_vot_mask2.ravel())
recall(mask.ravel(), maj_vot_mask2.ravel())
#%%
i=0
#%%
outputSample(samples.iloc[i], performance['Thr'])
i += 1
#%% EVALUATE PATCHES
#plt.imshow(img.masked_image)
#plt.show()

outcomes = []
#i=0
for patch in tqdm(img.patches, total=len(img.patches)):

#    i += 1
    patch_image = patch.patch_image
#    plt.imshow(patch_image)
    outcome, score, threshold = adModel.predict(patch_image)
    outcomes.append(outcome)
    patch.anomaly = outcome[1]
    patch.setScore(score, threshold)
#    patch.model = adModel.opt.name
    
#    print('Iter: {}/{}'.format(i, len(img.patches)))
#%%  
pathfile = csv_directory + 'result_test.csv'

if(os.path.exists(pathfile) == False):
    print('\n')
    print('Creation history test file ')
    history = pd.DataFrame()    
else:
    print('\n')
    print('Creation history test file ')
    history = pd.read_csv(pathfile)

#%%
simple_mask, n_anom = img.drawAnomaliesSimple() # 164592
history = history.append({'Image':index,'Method':'Simple','Stride_'+str(stride):evaluate(mask, simple_mask)}, ignore_index=True)

thr_all_mask, n_anom = img.drawAnomaliesThresholdindAll() # 71824
history = history.append({'Image':index,'Method':'Thr.All','Stride_'+str(stride):evaluate(mask, thr_all_mask)}, ignore_index=True)

#thr_map_mask = img.drawAnomaliesThesholding() # 71824
#history = history.append({'Image':index,'Method':'Thr.Map','Stride_'+str(stride):evaluate(mask, thr_map_mask)}, ignore_index=True)

maj_vot_mask = img.drawAnomaliesMajVoting() # 55396 8 
history = history.append({'Image':index,'Method':'Maj.vot.','Stride_'+str(stride):evaluate(mask, maj_vot_mask)}, ignore_index=True)# 57548 4

#history.to_csv('result_test.csv')

#%%
my_mask = simple_mask
                                       
error = my_mask == mask
error_value = 0
for rows in error:
    for col in rows:
        if(col == False):
            error_value += 1

error_value
#%%

pathfile = csv_directory + 'result_test.csv'

if(os.path.exists(pathfile) == False):
    history = pd.DataFrame({'Simple':[]})


#%% AUTOMATIC EVALUATION
filename = 'Ganom_v4.2_lr:0.0005|Epoch:32|Auc:0.879|Loss:204.7637.pth.tar'
adModel = loadModel(filename)
    
train = pd.read_csv(csv_directory + 'train_unique.csv', index_col=0)
shape = Shape(32,32)
stride = 4
start = 1000

automaticEvaluation(train, adModel, start)




















