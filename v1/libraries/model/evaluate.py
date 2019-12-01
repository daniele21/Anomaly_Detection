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

def evaluate(labels, scores, metric='roc', plot=False, folder_save=None):
    if metric == 'roc':
        return roc(labels, scores, plot, folder_save)
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

def roc(labels, scores, plot=False, folder_save=None):
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
    fpr, tpr, threshold = roc_curve(labels, scores)
    
    opt_threshold = _getOptimalThreshold(fpr, tpr, threshold)
    
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#    print(eer)
    
    if(plot):
        plt.figure()
        lw = 2
    #    plt.plot(0,0)
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.fill_between(fpr, tpr, alpha=0.3, color='orange')
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.plot(fpr, threshold, markeredgecolor='r',linestyle='dashed', color='r', label='Threshold')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
#        plt.show()

        print(folder_save)
        if(folder_save is not None):
            print('.. saving at {}'.format(folder_save))
            plt.savefig(folder_save + '/roc curve')
            
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





