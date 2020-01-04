# -*- coding: utf-8 -*-

from __future__ import print_function

#import os
from sklearn.metrics import roc_curve, auc, average_precision_score, recall_score
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from inspect import signature
from sklearn.metrics import confusion_matrix

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
#    print(confuseMatrix(y_true, y_pred))
    TN, FP, FN, TP = confuseMatrix(y_true, y_pred).ravel()
    return TP / (TP+FP)

def accuracy(y_true, y_pred):
    TN, FP, FN, TP = confuseMatrix(y_true, y_pred).ravel()
    return (TP + TN)/ (TP+FP+TN+FN)

def recall(y_true, y_pred):
    TN, FP, FN, TP = confuseMatrix(y_true, y_pred).ravel()
    return TP/ (TP+FN)

def IoU(pred_mask, true_mask):
#    print(pred_mask)
#    print(true_mask)
    SMOOTH = 1e-06
    
    intersection = pred_mask & true_mask
    union = pred_mask | true_mask
    
    iou = (intersection.sum() + SMOOTH) / (union.sum() + SMOOTH)
    
    return iou
    
def confuseMatrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def _getOptimalThreshold(fpr, tpr, threshold):
    #
    # tpr - (1-fpr) should be 0 or near to 0
    
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
    
    try:
        labels = labels.cpu()
        scores = scores.cpu()
    except:
        labels = labels
        scores = scores
    
#    print(scores)

    # True/False Positive Rates.
    
#    print('> Evaluation CHECK:')
#    print(labels.shape)
#    print(scores.shape)
    
    fpr, tpr, threshold = roc_curve(labels, scores)
    
    opt_threshold = _getOptimalThreshold(fpr, tpr, threshold)
    
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#    print(eer)
    
#    if(plot):
#        plt.figure()
#        lw = 2
#    #    plt.plot(0,0)
#        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
#        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
#        plt.fill_between(fpr, tpr, alpha=0.3, color='orange')
#        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
#        plt.plot(fpr, threshold, markeredgecolor='r',linestyle='dashed', color='r', label='Threshold = {:.3f}'.format(opt_threshold))
#        plt.xlim([0.0, 1.0])
#        plt.ylim([0.0, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.title('Receiver operating characteristic _{}_'.format(info))
#        plt.legend(loc="lower right")
##        plt.show()
#
#        print(folder_save)
#        if(folder_save is not None):
#            # SAVING ROC CURVE PLOT
#            print('.. saving at {}'.format(folder_save))
#            plt.savefig(folder_save + 'roc curve_' + str(info))
#            plt.show()
#            
#            # SAVING SCORES PLOT
#            plt.title('Anomaly Scores Trend _{}_'.format(info))
#            plt.plot(scores)
#            print('..{} saving at {}'.format(info, folder_save))
#            plt.savefig(folder_save + 'anomaly_scores_' + info + '_')
#            plt.show()
#        
#        
    
    if(plot):
        fig, [ax1, ax2] = plt.subplots(2,1, figsize=(8,12))
        
        lw = 2
        
        # PLOTTING AUC
        ax1.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        ax1.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        ax1.fill_between(fpr, tpr, alpha=0.3, color='orange')
        ax1.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        ax1.plot(fpr, threshold, markeredgecolor='r',linestyle='dashed', color='r', label='Threshold = {:.3f}'.format(opt_threshold))
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver operating characteristic _{}_'.format(info))
        ax1.legend(loc="lower right")
        
        # PLOTTING TREND SCORES
        ax2.set_title('Anomaly Scores Trend _{}_'.format(info))
        ax2.plot(scores)
        
        # SAVING PLOTS
        if(folder_save is not None):
            # SAVING ROC CURVE PLOT
            print('.. saving at {}'.format(folder_save))
            plt.savefig(folder_save + 'evaluation_' + str(info))
            plt.show()        

    return roc_auc, opt_threshold

def precision_recall(labels, scores, plot, folder_save):
    scores = scores.cpu()
    labels = labels.cpu()
    
    ap = average_precision_score(labels, scores)
    
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
#    return precision.shape, recall.shape, thresholds.shape
#    opt_threshold = _getOptimalThreshold(precision, recall, thresholds)
    
    if(plot):
        
        plt.fill_between(recall, precision, alpha=0.7, color='b')
        plt.axhline(ap, color='r', ls='--', label='Average Precision')
        plt.legend()
        #plt.plot(recall, precision)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap))
#        plt.show()
        
        if(folder_save is not None):
            print('.. saving at {}'.format(folder_save))
            plt.savefig(folder_save + '/prec-recall curve')
            
        plt.show()
        
    return ap






