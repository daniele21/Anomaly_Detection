# -*- coding: utf-8 -*-

#%%
import torch
import os
import pandas as pd
import numpy as np
from math import floor, ceil
#%% PATHS
class Paths():
    
    def __init__(self):
        self.curr_folder = os.getcwd()
        self.code_path = './'
#        self.dataset_path = '../../Dataset/my_dataset/'
#        self.dataset_path = './data/'
        self.dataset_path = '../dataset/'
        
        self.test_images = self.dataset_path + 'test_images/'
        self.test_labeled = self.test_images + 'labeled_images/'
        self.test_patched = self.test_images + 'patched_images/'
        
        self.library_folder = self.code_path + 'libraries/'
        self.csv_directory = '../data/'
        self.checkpoint_folder = '../../'
        self.dataloaders = '../../variables/v1/'
        
#        self.images_path = self.dataset_path + 'images/'
        self.images_path = '../../images/'
        self.patches_path = '../../patches/'
        self.normal_patches_path = self.patches_path + 'normals/'
        self.anom_patches_path = self.patches_path + 'anomalous/'
        

paths = Paths()
#%% CHECKPOINT
class Checkpoint():
    
#    def __init__(self, model, optimizer_gen, optimizer_discr, optimizer_weights,
#                 trainloss, validloss, folder_save, auc, threshold,
#                 scores, gt_labels, epoch, filename, save, opt):   
    def __init__(self, model):   
        
#        self.model = model
#        self.trainloss = trainloss
#        self.validloss = validloss
#        self.optimizer_gen = optimizer_gen
#        self.optimizer_discr = optimizer_discr
#        self.optimizer_weights = optimizer_weights
#        self.folder_save = folder_save
#        self.auc = auc
#        self.threshold = threshold
#        self.scores = scores
#        self.gt_labels = gt_labels
#        self.epoch = epoch
#        self.filename = filename
#        self.save = save   
#        
        self.train_loss = model.train_loss
        self.train_adv_loss = model.train_adv_loss
        self.train_con_loss = model.train_con_loss
        self.train_enc_loss = model.train_enc_loss
        
        self.valid_loss = model.val_loss
        self.valid_adv_loss = model.valid_adv_loss
        self.valid_con_loss = model.valid_con_loss
        self.valid_enc_loss = model.valid_enc_loss
        
        self.folder_save = model.folder_save
        self.auc = model.auc
        self.threshold = model.threshold
        self.scores = model.anomaly_scores
        self.gt_labels = model.gt_labels
        self.epoch = model.epoch
        self.opt = model.opt

    def saveCheckpoint(self, valid_loss):
        
        path_file = '{0}/CKP_{1}_lr:{2}|Epoch:{3}|Auc:{4:.3f}|Loss:{5:.4f}.pth.tar'.format(self.folder_save,
                                                                             self.opt.name,
                                                                             self.opt.lr_gen,
                                                                             self.epoch,
                                                                             self.auc,
                                                                             valid_loss)
        
        torch.save(self, path_file)
    

    def __repr__(self):
        return '-> Model: {}\n'.format(self.model) + \
                '-> Optimizer: {}\n'.format(self.optimizer) + \
                '-> Loss: {}\n'.format(self.validloss[-1]) + \
                '-> Epoch: {}\n'.format(self.epoch) + \
                '-> FilenName: {}'.format(self.filename)

class EarlyStopping():
    
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    
    """
    
    def __init__(self, patience, verbose=True, delta=0):
        """
        PARAMS:
            
            - patience : patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            - verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            - delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        print('\n-Early stopping Info:')
        score = -val_loss

        if self.best_score is None:
            print('>-Setting early stoppint')
            self.best_score = score
#            self.save_checkpoint(val_loss, ckp.model, ckp)
            self.save_checkpoint(val_loss)
            
            return True
            
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
            return False
        else:
            self.best_score = score
#            self.save_checkpoint(val_loss, ckp.model, ckp)
            self.save_checkpoint(val_loss)
            self.counter = 0
            
            return True
        
    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        
        if (self.verbose):
            print(f'> Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print('\n')
        
        self.val_loss_min = val_loss

class LR_decay():

    def __init__(self, lr):
        self.lr = lr

    def __call__(self, decay):
        self.lr = self.lr * decay
    
                     
#%%
#opt = Options()
#saveInfo(opt)
#%% FUNCTIONS

def computeAnomError(model, anom_set):
    anomalies = 0
    
    for image in anom_set:
        pred, _, _ = model.predict(image, 1, verbose=0)
#        print(label)
        anomalies += pred[1]
        
    error = anomalies / len(anom_set)
    error = 1 - error
    
    print('On {} anomalous images --> Error {:.3f}%'.format(len(anom_set), error*100))
    
    return error

def computeNormError(model, normal_set):
    normal = 0
    anomalies = 0
    
    for image in normal_set:
        pred, _, _ = model.predict(image, 0, verbose=0)
#        print(label)
        anomalies += pred[1]
    
    normal = len(normal_set) - anomalies
    
    error = normal / len(normal_set)
    error = 1 - error
    
    print('On {} normal images --> Error {:.3f}%'.format(len(normal_set), error*100))
    
    return error

def saveInfoAE(opt, path, auc):
    filename = opt.name + '_lr:{}'.format(opt.lr) + '_info'
    content = '\t\t' + opt.name + ' information\n\n'
    content = content + '- Dataset: {}\n'.format(opt.dataset)
    content = content + '- Anomaly at: {}%\n'.format(opt.anom_perc*100)
    content = content + '- Images: {}\n'.format(opt.nFolders)
    content = content + '- Patches per image: {}\n'.format(opt.patch_per_im)
    content = content + '- Patches: {}\n'.format(opt.nFolders*opt.patch_per_im)
    content = content + '- Channels: {}\n'.format(opt.in_channels)
    content = content + '- Split: {}-{:.2f}-{:.2f}\n'.format(opt.split, (1-opt.split)*0.5, (1-opt.split)*0.5)
    content = content + '- Lr: {}\n'.format(opt.lr)
    content = content + '- Batch_size: {}\n'.format(opt.batch_size)
    content = content + '- AUC: {:.2f}\n'.format(auc)
    content = content + '- Descr: {}\n'.format(opt.descr)
    
    f = open(path + filename, 'a')
    f.write(content)
    f.close()
    
def addInfoAE(opt, path, info):
    filename = opt.name + '_lr:{}'.format(opt.lr) + '_info'
    content = info
    
    f = open(path + filename, 'a')
    f.write(content)
    f.close()
    
def saveInfoGanomaly(opt, path, auc):
    filename = opt.name + '_lr:{}'.format(opt.lr_gen) + '_info'
    content = '\t\t' + opt.name + ' information\n\n'
    content = content + '- Dataset: {}\n'.format(opt.dataset)
#    content = content + '- Anomaly at: {}%\n'.format(opt.anom_perc*100)
    content = content + '- Patches per image: {}\n'.format(opt.patch_per_im)
    content = content + '- Patches: {}\n'.format(opt.nFolders*opt.patch_per_im)
    content = content + '- Images: {}\n'.format(opt.nFolders)
    content = content + '- Channels: {}\n'.format(opt.in_channels)
    content = content + '- Split: {}-{:.2f}-{:.2f}\n'.format(opt.split, (1-opt.split)*0.5, (1-opt.split)*0.5)
    content = content + '- Lr_Gen: {}\n'.format(opt.lr_gen)
    content = content + '- Lr_Discr: {}\n'.format(opt.lr_discr)
#    content = content + '- w_adv: {}\n'.format(opt.w_adv)
#    content = content + '- w_con: {}\n'.format(opt.w_con)
#    content = content + '- w_enc: {}\n'.format(opt.w_enc)
    content = content + '- Batch_size: {}\n'.format(opt.batch_size)
    content = content + '- AUC: {:.2f}\n'.format(auc)
    content = content + '- Descr: {}\n\n\n\n'.format(opt.descr)
    
    f = open(path + filename, 'a')
    f.write(content)
    f.close()
    
def addInfoGanomaly(opt, path, info):
    filename = opt.name + '_lr:{}'.format(opt.lr_gen) + '_info'
    content = info
    
    f = open(path + filename, 'a')
    f.write(content)
    f.close()
    
def writeDataResults(results, folder_save):
    filename = 'data_results'
#    content = '\t\t' + model_name + ' results\n\n'
    
    content = '- \t' + results['info'] + ':\n'
    content = content + '- Accuracy: \t{:.2f}\n'.format(results['acc'])
    content = content + '- Precision:\t{:.2f}\n'.format(results['prec'])
    content = content + '- Recall:   \t{:.2f}\n\n'.format(results['rec'])
#    
#    content = content + '- \tMajority Voting:\n'
#    content = content + '- Accuracy: \t{}\n'.format(maj_results['acc'])
#    content = content + '- Precision:\t{}\n'.format(maj_results['prec'])
#    content = content + '- Recall:   \t{}\n'.format(maj_results['rec'])
#    
    f = open(folder_save + filename, 'a')
    f.write(content)
    f.close()

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def check_existency(file):
    if(os.path.exists(file)):
        return True
    else:
        return False

def getAnomIndexes(targets):
    
    return np.where(targets)[0]
    
def getNmeans(array, n):
    new_array = []
    num_elem = len(array)
    
    slot_size = floor(num_elem / n)
#    rest = num_elem % n
#    print('\nSlot_size: ', slot_size)
    i = 0
    
    while(i <= num_elem-slot_size):
#        print(i)
        if(i + slot_size <= num_elem-slot_size):
#            print('if: ', i)
            mean = np.mean(array[i : i+slot_size])
            new_array.append(mean)
            i += slot_size
        else:
#            print('else: ', i)
            mean = np.mean(array[i : num_elem])
            new_array.append(mean)
            i += slot_size
    
    return np.array(new_array)
    

#%%
#a = [0,0,1, 1,2,2, 3,3,4, 4,5,5, 6,6,7, 7,8,8, 9] #18
##a.shape
#
#b = [1,1,3,2,2] #5
##b.shape
#
#c = getNmeans(a,2)
#d = getNmeans(b,1)
#
#print('\n')
#print(c)
#print('\n')
#print(d)
#d
