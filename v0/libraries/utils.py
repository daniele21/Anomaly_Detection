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
        self.dataset_path = '../../'
        
        self.test_images = self.dataset_path + 'test_images/'
        self.test_labeled = self.test_images + 'labeled_images/'
        self.test_patched = self.test_images + 'patched_images/'
        
        self.library_folder = self.code_path + 'libraries/'
        self.csv_directory = '../data/'
#        self.checkpoint_folder = self.dataset_path + 'checkpoints/'
        self.checkpoint_folder = self.dataset_path + '/'

        
#        self.images_path = self.dataset_path + 'images/'
        self.images_path = '../../images/'
        self.patches_path = '../../patches/'
        self.normal_patches_path = self.patches_path + 'normals/'
        self.anom_patches_path = self.patches_path + 'anomalous/'
        self.dataloaders_paths = self.dataset_path + 'dataloaders/'
        
        # ANOMALY IF 0.2 OF THE PATCH
        self.patched_images_dir = self.dataset_path + 'patched_images/'
        self.normal_images_path = self.dataset_path + 'patches/Normal_0.2/'
        self.anomalous_images_path = self.dataset_path + 'patches/Anomalous_0.2/'
        
        # ANOMALY IF 0.4 OF THE PATCH
        self.patched_images_dir_40 = self.dataset_path + 'patched_images_0.4/'
        self.normal_images_path_40 = self.dataset_path + 'patches/Normal_0.4/'
        self.anomalous_images_path_40 = self.dataset_path + 'patches/Anomalous_0.4/'
        
        # ANOMALY IF 0.5 OF THE PATCH
        self.patched_images_dir_50 = self.dataset_path + 'patched_images_0.5/'
        self.normal_images_path_50 = self.dataset_path + 'patches/Normal_0.5/'
        self.anomalous_images_path_50 = self.dataset_path + 'patches/Anomalous_0.5/'

paths = Paths()
#%% CHECKPOINT
class Checkpoint():
    
    
    def __init__(self, model, optimizer, trainloss, validloss, epoch,
                 filename, save):        
        self.model = model
        self.optimizer = optimizer
        self.trainloss = trainloss
        self.validloss = validloss
        self.epoch = epoch
        self.filename = filename
        self.save = save
    
    def _toTable(self):
        
        dfName = 'checkpoints.csv'
        path_file = paths.checkpoint_folder + dfName
        
        if(check_existency(path_file)):
            checkpoint = {'model'    : self.model,
                          'optimizer': self.optimizer,
                          'trainloss': self.trainloss,
                          'validloss': self.validloss,
                          'epoch'    : self.epoch,
                          'filename' : self.filename}
            
            df = pd.read_csv(path_file, index_col=0)
            df = df.append(checkpoint, ignore_index=True)
            df.to_csv(path_file)
            
        else:
            checkpoint = {'model'    : [self.model],
                          'optimizer': [self.optimizer],
                          'trainloss': [self.trainloss],
                          'validloss': [self.validloss],
                          'epoch'    : [self.epoch],
                          'filename' : [self.filename]}
            
            df = pd.DataFrame(checkpoint)
            df.to_csv(path_file)    

    def saveCheckpoint(self):
        if(self.save):
            torch.save(self, self.filename)
            self._toTable()

    

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
    
    def __init__(self, opt, verbose=True, delta=0):
        """
        PARAMS:
            
            - opt.patience : patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            - verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            - delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = opt.patience
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

def computeAnomError(model, anom_set, thr=None):
    anomalies = 0
    
    for image in anom_set:
        pred, _, _ = model.predict(image, threshold=thr, target=1, verbose=0)
#        print(label)
        anomalies += pred[1]
        
    error = anomalies / len(anom_set)
    error = 1 - error
    
    print('On {} anomalous images --> Error {:.3f}%'.format(len(anom_set), error*100))
    
    return error

def computeNormError(model, normal_set, thr=None):
    normal = 0
    anomalies = 0
    
    for image in normal_set:
        pred, _, _ = model.predict(image, threshold=thr, target=0, verbose=0)
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
    content = content + '- Anomaly at: {}%\n'.format(opt.anom_perc*100)
    content = content + '- Patches per image: {}\n'.format(opt.patch_per_im)
    content = content + '- Patches: {}\n'.format(opt.nFolders*opt.patch_per_im)
    content = content + '- Images: {}\n'.format(opt.nFolders)
    content = content + '- Channels: {}\n'.format(opt.in_channels)
    content = content + '- Split: {}-{:.2f}-{:.2f}\n'.format(opt.split, (1-opt.split)*0.5, (1-opt.split)*0.5)
    content = content + '- Lr_Gen: {}\n'.format(opt.lr_gen)
    content = content + '- Lr_Discr: {}\n'.format(opt.lr_discr)
    content = content + '- w_adv: {}\n'.format(opt.w_adv)
    content = content + '- w_con: {}\n'.format(opt.w_con)
    content = content + '- w_enc: {}\n'.format(opt.w_enc)
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
    content = content + '- Iou   :   \t{:.2f}\n\n'.format(results['iou'])
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

#        print(i)
#        print(new_array)
    
#    mean = np.mean(array[i:num_elem])
#    new_array.append(mean)
    
    return np.array(new_array)
