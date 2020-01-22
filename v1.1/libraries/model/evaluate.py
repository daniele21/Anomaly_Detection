# -*- coding: utf-8 -*-

from __future__ import print_function

#import os
from sklearn.metrics import roc_curve, auc, average_precision_score, recall_score
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.stats import cumfreq
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from inspect import signature
from sklearn.metrics import confusion_matrix
from libraries.utils import ensure_folder

EXTENSION = '.png'

def evaluate(labels, scores, metric='roc', plot=False, folder_save=None, info=''):
    if metric == 'roc':
        return roc(labels, scores, plot=plot, folder_save=folder_save, info=info)
    elif metric == 'avg_prec':
        return average_precision_score(labels, scores)
    elif metric == 'recall':
        return recall_score(labels, scores)
    elif metric == 'prec_rec_curve':
        return precision_recall(labels, scores, plot, folder_save)
    elif metric == 'precision':
#        return precision_score(labels, scores)
        return precision_score(labels, scores, average='macro')
    elif metric == 'acc':
        return accuracy_score(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

def precision(y_true, y_pred):
    TN, FP, FN, TP = confuseMatrix(y_true, y_pred).ravel()
    return TP / (TP+FP)

def accuracy(y_true, y_pred):
    TN, FP, FN, TP = confuseMatrix(y_true, y_pred).ravel()
    return (TP + TN)/ (TP+FP+TN+FN)

def recall(y_true, y_pred):
    TN, FP, FN, TP = confuseMatrix(y_true, y_pred).ravel()
    return TP/ (TP+FN)

def IoU(pred_mask, true_mask):
    true_mask = true_mask.astype(int)
    pred_mask = pred_mask.astype(int)

    SMOOTH = 1e-06
    
    intersection = pred_mask & true_mask
    union = pred_mask | true_mask
    
    iou = (intersection.sum() + SMOOTH) / (union.sum() + SMOOTH)
    
    return iou

def dice(pred_mask, true_mask):
    true_mask = true_mask.astype(int)
    pred_mask = pred_mask.astype(int)
    
    SMOOTH = 1e-06
    
    intersection = pred_mask & true_mask
    denom = pred_mask.sum() + true_mask.sum()
    
    dice_score = 2*(intersection.sum() + SMOOTH) / (denom + SMOOTH)
    
    return dice_score
    
def confuseMatrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def _getOptimalThreshold(fpr, tpr, threshold):
    
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i),
                        'threshold' : pd.Series(threshold, index=i)})
#    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
    roc_t = roc.loc[roc.tf.abs() == roc.tf.abs().min()]
    opt_threshold = roc_t.threshold
    
    return opt_threshold.values[0]

#%%

def roc(labels, scores, info='', plot=False, folder_save=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, tpr, threshold = roc_curve(labels, scores)
    
    opt_threshold = _getOptimalThreshold(fpr, tpr, threshold)
    
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
       
    
    if(plot):
        fig, [ax1, ax2, ax3] = plt.subplots(3,1, figsize=(8,10))
        
        lw = 2
        
        # PLOTTING AUC
        ax1.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.3f, EER = %0.3f)' % (roc_auc, eer))
        ax1.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        ax1.fill_between(fpr, tpr, alpha=0.3, color='orange')
        ax1.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        ax1.plot(fpr, threshold, markeredgecolor='r',linestyle='dashed', color='r', label='Threshold = {:.5f}'.format(opt_threshold))
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver operating characteristic _{}_'.format(info))
        ax1.legend(loc="lower right")
        
        # PLOTTING PRECISION-RECALL
        avg_prec = average_precision_score(labels, scores)
        precision, recall, thresholds = precision_recall_curve(labels, scores)
    
        ax2.fill_between(recall, precision, alpha=0.7, color='b')
        ax2.axhline(avg_prec, color='r', ls='--', label='Average Precision')
        ax2.legend()
        ax2.plot(recall, precision)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlim([0.0, 1.0])
        ax2.set_title('2-class Precision-Recall curve: Avg_Prec={0:0.2f}'.format(avg_prec))
    
        # PLOTTING TREND SCORES
        ax3.set_title('Anomaly Scores Trend _{}_'.format(info))
        ax3.hist(scores, bins=100)
        
        fig.tight_layout()        
        
        # SAVING PLOTS
        if(folder_save is not None):
            # SAVING ROC CURVE PLOT
            print('.. saving at {}'.format(folder_save))
            ensure_folder(folder_save)
            plt.savefig(folder_save + 'evaluation_' + str(info) + EXTENSION)
            plt.show()    
            

    print('> AUC {}      :\t{:.3f}'.format(str(info), roc_auc))
    print('> EER {}      :\t{:.3f}'.format(str(info), eer))
    print('> Threshold {}:\t{:.5f}\n'.format(str(info), opt_threshold))

    return roc_auc, opt_threshold

def precision_recall(labels, scores, plot=False, folder_save=None):
    
    try:
        labels = labels.cpu()
        scores = scores.cpu()
    except:
        labels = labels
        scores = scores
    
    avg_prec = average_precision_score(labels, scores)
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
#    return precision.shape, recall.shape, thresholds.shape
#    opt_threshold = _getOptimalThreshold(precision, recall, thresholds)
    
    if(plot):
        
        plt.fill_between(recall, precision, alpha=0.7, color='b')
        plt.axhline(avg_prec, color='r', ls='--', label='Average Precision')
        plt.legend()
        plt.plot(recall, precision)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: Avg_Prec={0:0.2f}'.format(avg_prec))
#        plt.show()
        
        if(folder_save is not None):
            print('.. saving at {}'.format(folder_save))
            plt.savefig(folder_save + '/prec-recall curve' + EXTENSION)
            
        plt.show()
        
    return avg_prec

#%%
def getThreshold(scores, prob, hist):
    
    values = hist[0]
    bins = hist[1]
    density = np.cumsum(values)/np.cumsum(values)[-1]
    
    index = np.where(density >= prob)[0][0]
    
    threshold = bins[index]
    
    return threshold

def evaluateRoc(scores, mask, info='', plot=True,
                thr=None, color='r', figsize=(8,10)):
    
    fpr, tpr, threshold = roc_curve(mask, scores)
    opt_threshold = _getOptimalThreshold(fpr, tpr, threshold)
    
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    if(plot):
        fig, [ax1, ax2, ax3] = plt.subplots(3,1, figsize=figsize)
        
        lw = 2
        
        # PLOTTING AUC
        ax1.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.3f, EER = %0.3f)' % (roc_auc, eer))
        ax1.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        ax1.fill_between(fpr, tpr, alpha=0.3, color='orange')
        ax1.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        ax1.plot(fpr, threshold, markeredgecolor='r',linestyle='dashed', color='r', label='Threshold = {:.5f}'.format(opt_threshold))
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver operating characteristic _{}_'.format(info.upper()))
        ax1.legend(loc="lower right")
        
        # PLOTTING PRECISION-RECALL
        avg_prec = average_precision_score(mask, scores)
        precision, recall, thresholds = precision_recall_curve(mask, scores)
    
        ax2.fill_between(recall, precision, alpha=0.7, color='b')
        ax2.axhline(avg_prec, color='r', ls='--', label='Average Precision')
        ax2.legend()
        ax2.plot(recall, precision)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlim([0.0, 1.0])
        ax2.set_title('2-class Precision-Recall curve: Avg_Prec={0:0.2f}'.format(avg_prec))
    
        # PLOTTING TREND SCORES
        ax3.set_title('Anomaly Scores Trend _{}_'.format(info))
        if(thr is None):
            thr = max(scores)/6
        
        n, bins, _ = ax3.hist(scores, bins=50, density=True, range=(0, thr*100/95))
        if(thr is not None):
            ax3.plot([thr, thr], [0, max(n)], c=color, label='Threshold' )

        ax3.plot([opt_threshold, opt_threshold], [0, max(n)], c='green', label='Best Threshold' )
        
        ax3.legend(loc='best')
        
        fig.tight_layout()
        
    print('> AUC {}      :\t{:.3f}'.format(str(info), roc_auc))
    print('> EER {}      :\t{:.3f}'.format(str(info), eer))
    print('> Threshold {}:\t{:.5f}\n'.format(str(info), opt_threshold))

    return roc_auc, opt_threshold



