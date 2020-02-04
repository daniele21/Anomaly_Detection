# -*- coding: utf-8 -*-

#%% IMPORTS
import numpy as np
import time
from PIL import Image
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as Transforms
import random
from tqdm import tqdm 
from copy import deepcopy
import sys
import math
import pandas as pd
from matplotlib import pyplot as plt

from libraries.utils import Paths, ensure_folder
from libraries.model import postprocessing as pp
from libraries.model import score

paths = Paths()

NORMAL_PATH = paths.normal_patches_path
ANOM_PATH = paths.anom_patches_path

NORMAL_LABEL = np.float64(0)
ANOMALY_LABEL = np.float64(1)
#%%
def applyMask(image, mask):
    
    masked_image = deepcopy(image)    
    masked_image[mask==1, 2] = 255
    
    return masked_image

def getImages(start, end):
    train = pd.read_csv(paths.csv_directory + 'train_unique.csv')
    
    images = []
    masks = []
    masked_images = []
    
    count = start
    
    for row in train.index[start : end]:
        print('Image n. {}'.format(count))
        filename    = train.iloc[row].Image_Id
        enc_pixels  = train.iloc[row].Encoded_Pixels
        
        img = cv2.imread(paths.images_path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = computeMask(enc_pixels, img)  
        masked_image = applyMask(img, mask)
        
        images.append(img)
        masks.append(mask)
        masked_images.append(masked_image)
        
        count += 1
    
    return images, masks, masked_images

def getImagesPerClass(n_samples):
    path_file = './libraries/dataset_package/'
    namefile = 'train_unique.csv'
    
    data = pd.read_csv(path_file + namefile, index_col=0)
    data1 = data.loc[data.Class_Id == '1']
    data2 = data.loc[data.Class_Id == '2']
    data3 = data.loc[data.Class_Id == '3']
    data4 = data.loc[data.Class_Id == '4']
    
    data = {1:data1,
            2:data2,
            3:data3,
            4:data4}
    
    images = {1:[],
              2:[],
              3:[],
              4:[]}
    
    masks = {1:[],
              2:[],
              3:[],
              4:[]}
    
    data_list = [data1, data2, data3, data4]
    defect = 1
    i = 0
    
    for i_data in data_list:
        i = 0
        for row in range(n_samples):
            index_file = i_data['index'].iloc[i]
            print('Image_n°{}: im_{}'.format(i, index_file))
            
            filename    = i_data.iloc[row].Image_Id
            enc_pixels  = i_data.iloc[row].Encoded_Pixels
    
            image = cv2.imread(paths.images_path + filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = computeMask(enc_pixels, image)
            
            images[defect].append(image)
            masks[defect].append(mask)
            
            i += 1
    
        defect += 1
        
    return images, masks

def getImagesFromSamples(samples):
    train = pd.read_csv(paths.csv_directory + 'train_unique.csv')
    
    images = []
    masks = []
    masked_images = []
    
    count = 0
    
    print('\n\nImage n. {} '.format(len(samples)))
    
    for row in samples:
        
        filename    = train.iloc[row].Image_Id
        enc_pixels  = train.iloc[row].Encoded_Pixels
        
        img = cv2.imread(paths.images_path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = computeMask(enc_pixels, img)  
        masked_image = applyMask(img, mask)
        
        images.append(img)
        masks.append(mask)
        masked_images.append(masked_image)
        
        count += 1
    
    return images, masks, masked_images

def getPatchesFromImages(start, end, shape):
    train = pd.read_csv(paths.csv_directory + 'train_unique.csv')
    
    patches_list = []
    patch_masks_list = []
    
    count = start
#    print(len(train.index[start : end]))
    for row in train.index[start : end]:
        print('Image n. {}'.format(count))
        filename    = train.iloc[row].Image_Id
        enc_pixels  = train.iloc[row].Encoded_Pixels
        
        img = cv2.imread(paths.images_path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = computeMask(enc_pixels, img)    
        
        patches, patch_masks = _splitPatches(img, mask, shape)
        
        patches_list.extend(patches)
        patch_masks_list.extend(patch_masks)
        
        count += 1
    
    return patches_list, patch_masks_list
    
def _splitPatches(image, mask, shape):
    
    x = image.shape[1]
    y = image.shape[0]
    
    patches = []
    patch_masks = []
    
    # 400 patches per image
    for j in range(0, y, shape):
        for i in range(0, x, shape):
            if(j+shape < y and i+shape < x):
                patch = image[j:j+shape, i:i+shape]
    #            print('{}x{}'.format(j,i))
                patches.append(patch)
                
                patch_mask = mask[j:j+shape, i:i+shape]
                patch_masks.append(patch_mask)
            
    return patches, patch_masks
    
    
def computeMask(enc_pixel, img):
    width = img.shape[0]
    height= img.shape[1]
    
    mask= np.zeros( width*height ).astype(np.float64)
    if(enc_pixel == 0):
        return np.zeros((width, height))
    
    array = np.asarray([int(x) for x in enc_pixel.split()])
    starts = array[0::2]
    lengths = array[1::2]
    
    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    mask = np.flipud(np.rot90(mask.reshape(height,width), k=1))
    mask = mask.astype(np.float64)
    
    return mask

def _setupDataset(opt, train='normal', valid='normal', test='mixed'):
    '''
        train:      'normal' / 'mixed'
        validation: 'normal' / 'mixed'
        test:       'normal' / 'mixed'
    
    '''
    patch_x_img = opt.patch_per_im
    n_images = opt.nFolders
    n_patches = n_images * patch_x_img
    
    n_training_imgs = math.ceil(n_patches * opt.split)
    n_validation_imgs = math.ceil(n_patches * (1-opt.split))
    n_test_imgs = math.ceil(n_patches * (1-opt.split))
    
#    print(n_training_imgs)
#    print(n_validation_imgs)
#    print(n_test_imgs)
    
    training_set    = {'DATA':[], 'LABELS':[]}
    validation_set  = {'DATA':[], 'LABELS':[]}
    test_set        = {'DATA':[], 'LABELS':[]}
    
    start = time.time()
    
    n_anom_patches = int((n_training_imgs + n_validation_imgs + n_test_imgs)/2)
    anom_patches = createAnomalousPatches(n_anom_patches)
    
    n_norm_patches = int(n_training_imgs + n_validation_imgs + n_test_imgs)
    norm_patches = createNormalPatches(n_norm_patches)
                    
    # TRAINING SET
    training_set = fillSet(train, norm_patches, anom_patches, int(n_training_imgs))
    
    # VALIDATION
    validation_set = fillSet(train, norm_patches[int(n_training_imgs):],
                                    anom_patches[int(n_training_imgs//2) : ],
                                    
                                    int(n_validation_imgs))
    
    # TEST
    test_set = fillSet(train, norm_patches[int(n_training_imgs + n_validation_imgs):],
                              anom_patches[int(n_training_imgs//2) + int(n_validation_imgs//2): ],
                              
                              int(n_validation_imgs))
        

    print('\n')
    print('Training set:   {} images'.format(len(training_set['DATA'])))
    print('Validation set: {} images'.format(len(validation_set['DATA'])))
    print('Test set:       {} images'.format(len(test_set['DATA'])))
    
    end = time.time()
    print('Spent time : {:.2f} sec'.format(end - start))
    
    return training_set, validation_set, test_set

 
def fillSet(set_type, norm_patches, anom_patches, n_images):
    '''
    DESCRIPTION:
            Automatic dataset filling for training, validation and testing sets
    
    PARAMS:
            set_type : 'normal' / 'mixed'
    
    '''
    dataset = {'DATA':[], 'LABELS':[]}
#    anoms = 0
    
    if(set_type == 'normal'):
        # NORMALS
        dataset['DATA'] = norm_patches[0:n_images]
        dataset['LABELS'] = np.zeros([n_images]).fill(NORMAL_LABEL)
    
    elif(set_type == 'mixed'):
        # NORMALS
        dataset['DATA'] = norm_patches[0:int(n_images//2)]
        dataset['LABELS'] = np.zeros([n_images//2])
        dataset['LABELS'].fill(NORMAL_LABEL)
        
        # ANOMALOUS
        dataset['DATA'] = np.concatenate((dataset['DATA'], anom_patches[0:int(n_images//2)]))
        temp = np.zeros([n_images//2])
        temp.fill(ANOMALY_LABEL)
        dataset['LABELS'] = np.concatenate((dataset['LABELS'], temp))
            
    return dataset

def createAnomalousPatches(N):
    
    anom_patches = []
    i=0
    for index in os.listdir(ANOM_PATH):
        if(len(anom_patches) >= N):
            break
        
        path_anom = ANOM_PATH + str(index) + '/'
        anom_filename = os.listdir(path_anom)
        
        for filename in anom_filename:
            if(len(anom_patches) < N):
                image = cv2.imread(path_anom + filename)           
                anom_patches.append(image)
                i += 1
#                print(i)
            else:
                break
            
    random.shuffle(anom_patches)
    return anom_patches

def createNormalPatches(N):
    
    norm_patches = []
    i=0
    for index in os.listdir(NORMAL_PATH):
        if(len(norm_patches) >= N):
            break
        
        path_norm = NORMAL_PATH + str(index) + '/'
        norm_filename = os.listdir(path_norm)
        
        for filename in norm_filename:
            if(len(norm_patches) < N):
                image = cv2.imread(path_norm + filename)           
                norm_patches.append(image)
                i += 1
#                print(i)
            else:
                break
            
    random.shuffle(norm_patches)
    return norm_patches
        

def loadDataset(opt, test='mixed'):
    '''
    DESCRIPTION:
        Load data in train, validation and test set, in the following way
        
        - Training set:     70 %   NORMAL SAMPLES
        - Validation set:   15 %   NORMAL SAMPLES
        - Test set:         15 %   NORMAL SAMPLES(test='normal') / NORMAL-ANOM(test='mixed')
    
    '''
    training_set    = {'DATA':[], 'LABELS':[]}
    validation_set  = {'DATA':[], 'LABELS':[]}
    test_set        = {'DATA':[], 'LABELS':[]}
    
    patches_per_image = opt.patch_per_im
    
    NORMAL_LABEL = np.float64(0)
    ANOMALY_LABEL = np.float64(1)
    
    training_index = int(patches_per_image * opt.split)
    validation_index = training_index + int(patches_per_image*(1-opt.split))
    n_test_samples = int(training_index * (1-opt.split))
#    print(training_index)
#    print(validation_index)
#    print(n_test_samples)
    
    counter = 0
    start = time.time()
    
    folders = os.listdir(NORMAL_PATH)
    random_folders = random.sample(folders, opt.nFolders)
    
    print('\nPatches per image: ', patches_per_image)
    for index in random_folders:
        counter += 1
        print('\n')
        print('Image n.', counter)
        
        path_normal = NORMAL_PATH + str(index) + '/'
        path_anom = ANOM_PATH + str(index) + '/'
        
        norm_filename = os.listdir(path_normal)
        random.shuffle(norm_filename)
        
        anom_filename = os.listdir(path_anom)
        random.shuffle(anom_filename)
                    
        # TRAINING SET
        for filename in tqdm(norm_filename[0 : training_index], leave=True, desc='Training-set\t', file=sys.stdout):

            image = cv2.imread(path_normal + filename)
           
#            if(channels == 1):
#                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           

#            train_data.append(image)
#            train_label.append(NORMAL_LABEL)
            training_set['DATA'].append(image)
            training_set['LABELS'].append(NORMAL_LABEL)
        
        # VALIDATION
        for filename in tqdm(norm_filename[training_index : validation_index], leave=True, desc='Validation-set\t',
                             file=sys.stdout):
            
            image = cv2.imread(path_normal + filename)
            
#            if(channels == 1):
#                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           
#             validation_data.append(image)
#            validation_label.append(NORMAL_LABEL)
            validation_set['DATA'].append(image)
            validation_set['LABELS'].append(NORMAL_LABEL)
            
            if(test=='mixed'):
                test_set['DATA'].append(image)
                test_set['LABELS'].append(NORMAL_LABEL)

        # TEST
        if(test=='mixed'):
            test_filename = anom_filename[0:len(validation_set['DATA'])]
            LABEL = ANOMALY_LABEL
#            test_set = deepcopy(validation_set)
            path_images = path_anom
            
        elif(test=='normal'):
            test_filename = norm_filename[validation_index : validation_index+n_test_samples]
            LABEL = NORMAL_LABEL
            path_images = path_normal
            
        else:
            raise Exception('>>> ERROR Load Dataset: wrong test_type   <<<')
            
        for filename in tqdm(test_filename, leave=True, desc='Test-set\t', file=sys.stdout):
#            print(path_anom + filename)
            image = cv2.imread(path_images + filename)
           
#            if(channels == 1):
#                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           
#            print(image)
#            test_data.append(image)
#            test_label.append(ANOM_PATH)
            test_set['DATA'].append(image)
            test_set['LABELS'].append(LABEL)

    print('\n')
    print('Training set:   {} images'.format(len(training_set['DATA'])))
    print('Validation set: {} images'.format(len(validation_set['DATA'])))
    print('Test set:       {} images'.format(len(test_set['DATA'])))
    
    end = time.time()
    print('Spent time : {:.2f} sec'.format(end - start))
    
    return training_set, validation_set, test_set

#%%
    
def generateDataloader(opt):
    print('\n>Loading Steel Dataset')
    
    if(opt.loadedData == False):
        opt.loadDatasets()
    
    dataset = {}
    dataset['train']       = SteelDataset(opt, train=True)
    dataset['validation']  = SteelDataset(opt, valid=True)
    dataset['test']        = SteelDataset(opt, test=True)
    
    shuffle = {'train':True, 'validation':True, 'test':True}
    
    dataloader = {x: DataLoader(dataset    = dataset[x],
                                batch_size = opt.batch_size,
                                drop_last  = True,
                                shuffle = shuffle[x],
                                num_workers= opt.n_workers
                                )
                  
                  for x in ['train', 'validation', 'test']}
    
    return dataloader

def generateDataloaderTL(opt):
    print('\n>Loading Steel Dataset')
    
    if(opt.loadedData == False):
        opt.loadDatasets()
    
    dataset = {}
    dataset['train']       = TLSteelDataset(opt, train=True)
    dataset['validation']  = TLSteelDataset(opt, valid=True)
    dataset['test']        = TLSteelDataset(opt, test=True)
    
    shuffle = {'train':True, 'validation':True, 'test':True}
    
    dataloader = {x: DataLoader(dataset    = dataset[x],
                                batch_size = opt.batch_size,
                                drop_last  = True,
                                shuffle = shuffle[x],
                                num_workers= opt.n_workers
                                )
                  
                  for x in ['train', 'validation', 'test']}
    
    return dataloader

def generateDataloaderAS_simple(opt, dataset):
    
    data = ASDataset(dataset)
    
#    data = {}  
#    data = DefectDataset(as_scores, masks, opt)
    
    dataloader = DataLoader(dataset = data,
                            batch_size = opt.batch_size,
                            drop_last  = True,
                            num_workers= opt.n_workers
                            )
    
    return dataloader

def generateDataloaderAS(opt, adModel, samples, stride=8):
    
    dataset = {}
    dataset['DATA'], dataset['LABELS'], _ = getImagesFromSamples(samples)
    dataset['SAMPLES'] = samples

    as_scores = []
    masks = []
    
    for i in tqdm(range(len(dataset['DATA'])), total=len(dataset['DATA'])):
#        print(i)
        as_score, mask = score.anomalyScoreFromImage(adModel, dataset['DATA'][i],
                                                     dataset['LABELS'][i],
                                                               stride, 32)
        as_scores.append(as_score)  
        masks.append(mask)
        
    dataset['AS'] = as_scores
    dataset['LABELS'] = masks
    
    data = ASDataset(dataset)
    
#    data = {}  
#    data = DefectDataset(as_scores, masks, opt)
    
    dataloader = DataLoader(dataset = data,
                            batch_size = opt.batch_size,
                            drop_last  = True,
                            num_workers= opt.n_workers
                            )
    
    return dataloader

def generateDataloaderPerDefect(opt, adModel, samples, stride=8):
    
    images, targets, masked_images = getImagesFromSamples(samples)
 
    as_scores = []
    masks = []
    
    dataset = {}
    
    j=0
    for image in tqdm(images, total=len(images)):
        as_score, mask = score.anomalyScoreFromImage(adModel, image, targets[j],
                                                               stride, 32)
        as_scores.append(as_score)
        masks.append(mask)
        j+=1
        
        
    dataset = DefectDataset(as_scores, masks, opt)
    
    dataloader = DataLoader(dataset = dataset,
                            batch_size = opt.batch_size,
                            drop_last  = True,
                            num_workers= opt.n_workers
                            )
    
    return dataloader


def generateDataloaderFromDatasets(opt, training_set, validation_set, test_set):
    print('\n>Loading Steel Dataset')
    
    dataset = {}
    dataset['train']       = SteelDataset(opt, training_set, train=True)
    dataset['validation']  = SteelDataset(opt, validation_set, valid=True)
    dataset['test']        = SteelDataset(opt, test_set, test=True)
    
    shuffle = {'train':True, 'validation':True, 'test':True}
    
    dataloader = {x: DataLoader(dataset    = dataset[x],
                                batch_size = opt.batch_size,
                                drop_last  = True,
                                shuffle = shuffle[x],
                                num_workers= opt.n_workers
                                )
                  
                  for x in ['train', 'validation', 'test']}
    
    return dataloader

def generateDataloaderTest(patches, opt):
    
    data = []
    targets = []
    
    
    for patch in patches:
        data.append(patch)
        targets.append(np.float64(patch.anomaly))
    
    dataset = TestDataset(data, targets, opt)
    
    dataloader = DataLoader(dataset,
                            batch_size = opt.batch_size,
#                            shuffle=True,
                            drop_last  = True,
                            num_workers = 8)
    
    return dataloader

def dataloaderPatchMasks(opt):
    
    dataset = {}
    dataset['DATA'], dataset['LABELS'] = getPatchesFromImages(opt.start, opt.end, opt.shape)
    
    training_set = {}
    validation_set = {}
    test_set = {}
    
    train_index = int(len(dataset['DATA']) * opt.split)
    valid_index = int(len(dataset['DATA']) * (0.9-opt.split))
    test_index = int(len(dataset['DATA']) * 0.1)
    
#    print(train_index)
#    print(valid_index)
#    print(test_index)
    
    start = 0
    end = train_index
    training_set['DATA'] = dataset['DATA'][start:end]
    training_set['LABELS'] = dataset['LABELS'][start:end]
    
    start = train_index
    end = train_index + valid_index
    validation_set['DATA'] = dataset['DATA'][start:end]
    validation_set['LABELS'] = dataset['LABELS'][start:end]
    
    start = train_index + valid_index 
    end = train_index + valid_index + test_index
    test_set['DATA'] = dataset['DATA'][start : end]
    test_set['LABELS'] = dataset['LABELS'][start : end]
    
    dataset = {}
    dataset['train']       = SteelDataset(opt, training_set, train=True)
    dataset['validation']  = SteelDataset(opt, validation_set, valid=True)
    dataset['test']        = SteelDataset(opt, test_set, test=True)
    
    shuffle = {'train':True, 'validation':True, 'test':True}
    
    dataloader = {x: DataLoader(dataset    = dataset[x],
                                batch_size = opt.batch_size,
                                drop_last  = True,
                                shuffle = shuffle[x],
                                num_workers= opt.n_workers
                                )
                  
                  for x in ['train', 'validation', 'test']}
    
    return dataloader

def dataloaderSingleSet(samples, batch_size):
    
    dataset = {}
    dataset['DATA'], dataset['LABELS'], dataset['MASKED'] = getImagesFromSamples(samples)
    dataset['SAMPLES'] = samples
    
    dataset = ImagesDataset(dataset)

    dataloader = DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            drop_last  = True,
#                            shuffle = shuffle[x],
                            num_workers= 8
                            )
    
    return dataloader

def dataloaderFullImages(opt):
    
    dataset = {}
    dataset['DATA'], dataset['LABELS'], _ = getImages(opt.start, opt.end)
    
    training_set = {}
    validation_set = {}
    test_set = {}
    
    train_index = int(len(dataset['DATA']) * opt.split)
    valid_index = int(len(dataset['DATA']) * (0.9-opt.split))
    test_index = int(len(dataset['DATA']) * 0.1)
    
    start = 0
    end = train_index
    training_set['DATA'] = dataset['DATA'][start:end]
    training_set['LABELS'] = dataset['LABELS'][start:end]
    
    start = train_index
    end = train_index + valid_index
    validation_set['DATA'] = dataset['DATA'][start:end]
    validation_set['LABELS'] = dataset['LABELS'][start:end]
    
    start = train_index + valid_index 
    end = train_index + valid_index + test_index
    test_set['DATA'] = dataset['DATA'][start : end]
    test_set['LABELS'] = dataset['LABELS'][start : end]
    
    dataset = {}
    dataset['train']       = FullSteelDataset(opt, training_set, train=True)
    dataset['validation']  = FullSteelDataset(opt, validation_set, valid=True)
    dataset['test']        = FullSteelDataset(opt, test_set, test=True)
    
    shuffle = {'train':True, 'validation':True, 'test':True}
    
    dataloader = {x: DataLoader(dataset    = dataset[x],
                                batch_size = opt.batch_size,
                                drop_last  = True,
                                shuffle = shuffle[x],
                                num_workers= opt.n_workers
                                )
                  
                  for x in ['train', 'validation', 'test']}
    
    return dataloader

def saveDataloader(dataloader):
    
    # CHECK EXISTENCY FOLDERS
    
    # TRAINING SET
    ensure_folder(paths.training_set_path)
    # VALIDATION SET
    ensure_folder(paths.validation_set_path)
    # TEST SET
    ensure_folder(paths.test_set_path)
    
    folders = {'train':paths.training_set_path,
               'validation':paths.validation_set_path,
               'test':paths.test_set_path}
    
    sets = ['train', 'validation', 'test']
    
    for i in range(len(dataloader['train'].dataset.data)):
        
        for set_type in sets:
            images = dataloader[set_type].dataset.data
            targets= dataloader[set_type].dataset.targets
            
            if(i < len(images)):
                images[i]
    
    
def collectNormalSamples(nImages, normal_per_img=None):
    n_normal = 0
    counterImage = 0
    limit = 0

    normalTest = []
    
    folders = os.listdir(NORMAL_PATH)
    random_folders = random.sample(folders, nImages)
    
    for index in random_folders:
        counterImage += 1
        path_anom = NORMAL_PATH + str(index) + '/'
        normImages = os.listdir(path_anom)
        
        print('\nImage n.', counterImage)
        
        if(normal_per_img is None):
            normal_per_img = len(normImages)
        
        limit = 0
#        print(len(normImages))
        for filename in tqdm(normImages, leave=True, desc='Anom. Images:\t', 
                             total=normal_per_img, file=sys.stdout):
            
            cond1 = normal_per_img is not None and limit < normal_per_img
            cond2 = normal_per_img is None
            
            if(cond1 or cond2):
                n_normal += 1
                limit += 1
                image = cv2.imread(path_anom + filename)
                normalTest.append(image)
                
        
            
    
    print('> {} anomalous images loaded'.format(n_normal))
    
    return normalTest
    
    
def collectAnomalySamples(nImages, anom_per_img=None):
    n_anomaly = 0
    counterImage = 0
    limit = 0

    anomalyTest = []
    
    folders = os.listdir(ANOM_PATH)
    random_folders = random.sample(folders, nImages)
    
    for index in random_folders:
        counterImage += 1
        path_anom = ANOM_PATH + str(index) + '/'
        anomImages = os.listdir(path_anom)
        
        print('\nImage n.', counterImage)
        
        if(anom_per_img is None):
            anom_per_img = len(anomImages)
        
        limit = 0
        
        for filename in tqdm(anomImages, leave=True, desc='Anom. Images:\t', 
                             total=anom_per_img, file=sys.stdout):
            
            cond1 = anom_per_img is not None and limit < anom_per_img
            cond2 = anom_per_img is None
            
            if(cond1 or cond2):
                n_anomaly += 1
                limit += 1
                image = cv2.imread(path_anom + filename)
                anomalyTest.append(image)
            else:
                pass
    
    print('> {} anomalous images loaded'.format(n_anomaly))
    
    return anomalyTest


class DefectDataset(Dataset):
    
    def __init__(self, data, targets, opt):
        self.data = data
        self.targets = targets
        
        self.initTransform(opt)
        
    def initTransform(self, opt):
        self.transforms = Transforms.Compose(
                            [
#                                    transforms.Resize(32, interpolation=Image.BILINEAR),
                                Transforms.Grayscale(num_output_channels = opt.in_channels),
                                Transforms.ToTensor(),
                                Transforms.Normalize((0.5,),
                                                     (0.5,)),
#                                    Transforms.Grayscale(num_output_channels = opt.in_channels)
#                                    transforms.ToPILImage()
                            ]
                        )
    def __getitem__(self, index):
        
        
        
        if torch.is_tensor(index):
            index = index.tolist()
            
        image, target = self.data[index], self.targets[index]
#        print(image)
        image = Image.fromarray(image)
        
        image = self.transforms(image)
#        print(image)
        return image, target
    
    def __len__(self):
        return len(self.data)
    
class TestDataset(Dataset):
    
    def __init__(self, data, targets, opt):
        self.data = data
        self.targets = targets
        self._initTransforms(opt)
        
    def _initTransforms(self, opt):
        self.transforms = Transforms.Compose(
                            [
#                                    transforms.Resize(32, interpolation=Image.BILINEAR),
                                Transforms.Grayscale(num_output_channels = opt.in_channels),
                                Transforms.ToTensor(),
                                Transforms.Normalize((0.5,),
                                                     (0.5,)),
#                                    Transforms.Grayscale(num_output_channels = opt.in_channels)
#                                    transforms.ToPILImage()
                            ]
                        )
    
    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()
            
        image, target = self.data[index].patch_image, self.targets[index]
        image = Image.fromarray(image)
        
        # GRAYSCALE
#        image = image.convert('LA')
           
        image = self.transforms(image)
#        print(image.shape)
        return image, target
    
    def __len__(self):
        return len(self.data)
        
class TLSteelDataset(Dataset):
    
    def __init__(self, opt, dataset=None, train=False, valid=False, test=False):
        
        self.train = train
        self.valid = valid
        self.test = test
        
        if(dataset is None):
            if(train and not valid and not test):
                self.data = opt.training_set['DATA']
                self.data_TL = opt.training_set['DATA']
                self.targets = opt.training_set['LABELS']
            
            elif(valid and not train and not test):
                self.data = opt.validation_set['DATA']
                self.data_TL = opt.training_set['DATA']
                self.targets = opt.validation_set['LABELS']
            
            elif(test and not train and not valid):
                self.data = opt.test_set['DATA']
                self.data_TL = opt.training_set['DATA']
                self.targets = opt.test_set['LABELS']
                
        else:
            if(train and not valid and not test):
                self.data = dataset['DATA']
                self.data_TL = opt.training_set['DATA']
                self.targets = dataset['LABELS']
            
            elif(valid and not train and not test):
                self.data = dataset['DATA']
                self.data_TL = opt.training_set['DATA']
                self.targets = dataset['LABELS']
            
            elif(test and not train and not valid):
                self.data = dataset['DATA']
                self.data_TL = opt.training_set['DATA']
                self.targets = dataset['LABELS']
                
        self.data = np.vstack(self.data).reshape(-1, opt.shape, opt.shape, 3)
        self.data_TL = np.vstack(self.data_TL).reshape(-1, opt.shape, opt.shape, 3)

        print(self.data.shape)
        print(opt.augmentation)
        self.transforms = self.transf_small(opt)
        self.transforms_TL = self.transf_to_tl_size(opt)

    def transf_to_tl_size(self, opt):
        if(self.train):
            if(opt.augmentation==False):
                    
                    transforms = Transforms.Compose(
                                        [
        #                                    transforms.Resize(32, interpolation=Image.BILINEAR),
                                            Transforms.Grayscale(num_output_channels = opt.in_channels),
                                            Transforms.Resize((opt.TL_size,opt.TL_size)),
                                            Transforms.ToTensor(),
                                            
                                            Transforms.Normalize((0.5,),
                                                                 (0.5,)),
        #                                    Transforms.Grayscale(num_output_channels = opt.in_channels)
        #                                    transforms.ToPILImage()
                                        ]
                                    )
            else:
                    transforms = Transforms.Compose(
                                        [
                                            Transforms.Resize((opt.TL_size,opt.TL_size)),
                                            # AUGMENTATION
                                            Transforms.ColorJitter(brightness=0.2,
                                                                    contrast=0.2, 
                                                                    saturation=0.3, 
                                                                    hue=0.2),
                                            
        #                                    Transforms.RandomRotation(10),
                                            Transforms.RandomAffine(degrees=10,
                                                                    scale=(1,2),
                                                                    shear=0.2),
                                            
                                            Transforms.Grayscale(num_output_channels = opt.in_channels),
#                                            Transforms.Resize((224,224)),
                                            Transforms.ToTensor(),
                                            Transforms.Normalize((0.5,),
                                                                 (0.5,))   
                                                
                                        ]
                                    )  
                        
        elif(self.valid or self.test):
            transforms = Transforms.Compose(
                                    [
                                        Transforms.Grayscale(num_output_channels = opt.in_channels),
                                        Transforms.Resize((opt.TL_size,opt.TL_size)),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,),
                                                             (0.5,))  
                                    ]
                                )
        
        return transforms
    
    def transf_small(self, opt):
        if(self.train):
#            print(opt)
            if(opt.augmentation==False):
                    
                    transforms = Transforms.Compose(
                                        [
        #                                    transforms.Resize(32, interpolation=Image.BILINEAR),
                                            Transforms.Grayscale(num_output_channels = opt.in_channels),
    #                                        Transforms.Resize((224,224)),
                                            Transforms.ToTensor(),
                                            
                                            Transforms.Normalize((0.5,),
                                                                 (0.5,)),
        #                                    Transforms.Grayscale(num_output_channels = opt.in_channels)
        #                                    transforms.ToPILImage()
                                        ]
                                    )
            else:
                    transforms = Transforms.Compose(
                                        [
                                            # AUGMENTATION
                                            Transforms.ColorJitter(brightness=0.2,
                                                                    contrast=0.2, 
                                                                    saturation=0.3, 
                                                                    hue=0.2),
                                            
        #                                    Transforms.RandomRotation(10),
                                            Transforms.RandomAffine(degrees=10,
                                                                    scale=(0.5,2),
                                                                    shear=0.2),
                                            
                                            Transforms.Grayscale(num_output_channels = opt.in_channels),
    #                                        Transforms.Resize((224,224)),
                                            Transforms.ToTensor(),
                                            Transforms.Normalize((0.5,),
                                                                 (0.5,))   
                                                
                                        ]
                                    )  
                            
        
        elif(self.valid or self.test):
            transforms = Transforms.Compose(
                                    [
                                        Transforms.Grayscale(num_output_channels = opt.in_channels),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,),
                                                             (0.5,))  
                                    ]
                                )
        
        return transforms
    
    def __getitem__(self, index):
       
        if torch.is_tensor(index):
            index = index.tolist()
            
        image, image_TL, target = self.data[index], self.data_TL[index], self.targets[index]

#        image = Image.convert('RGB')
        image = Image.fromarray(image)
        image_TL = Image.fromarray(image_TL)
        
        # GRAYSCALE
#        image = image.convert('LA')
           
        image = self.transforms(image)
        image_TL = self.transforms_TL(image_TL)
        
        return image, image_TL, target
    
    def __len__(self):
        return len(self.data)       

class SteelDataset(Dataset):
    
    def __init__(self, opt, dataset=None, train=False, valid=False, test=False):
        
        self.train = train
        self.valid = valid
        self.test = test
        
        if(dataset is None):
            if(train and not valid and not test):
                self.data = opt.training_set['DATA']
                self.targets = opt.training_set['LABELS']
            
            elif(valid and not train and not test):
                self.data = opt.validation_set['DATA']
                self.targets = opt.validation_set['LABELS']
            
            elif(test and not train and not valid):
                self.data = opt.test_set['DATA']
                self.targets = opt.test_set['LABELS']
                
        else:
            if(train and not valid and not test):
                self.data = dataset['DATA']
                self.targets = dataset['LABELS']
            
            elif(valid and not train and not test):
                self.data = dataset['DATA']
                self.targets = dataset['LABELS']
            
            elif(test and not train and not valid):
                self.data = dataset['DATA']
                self.targets = dataset['LABELS']
                
        self.data = np.vstack(self.data).reshape(-1, opt.shape, opt.shape, 3)

        print(self.data.shape)

        self.transforms = self._initTransforms(opt)

    def _initTransforms(self, opt):
        if(self.train):
            if(opt.augmentation==False):
                
                transforms = Transforms.Compose(
                                    [
    #                                    transforms.Resize(32, interpolation=Image.BILINEAR),
                                        Transforms.Grayscale(num_output_channels = opt.in_channels),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,),
                                                             (0.5,)),
    #                                    Transforms.Grayscale(num_output_channels = opt.in_channels)
    #                                    transforms.ToPILImage()
                                    ]
                                )
                                
            else:
                transforms = Transforms.Compose(
                                    [
                                        # AUGMENTATION
                                        Transforms.ColorJitter(brightness=0.2,
                                                                contrast=0.2, 
                                                                saturation=0.3, 
                                                                hue=0.2),
                                        
    #                                    Transforms.RandomRotation(10),
                                        Transforms.RandomAffine(degrees=10,
                                                                scale=(0.5,2),
                                                                shear=0.2),
                                        
                                        Transforms.Grayscale(num_output_channels = opt.in_channels),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,),
                                                             (0.5,))   
                                            
                                    ]
                                )                                
                                
        elif(self.valid or self.test):
            transforms = Transforms.Compose(
                                    [
                                        Transforms.Grayscale(num_output_channels = opt.in_channels),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,),
                                                             (0.5,))  
                                    ]
                                )
        
        return transforms
    
    def __getitem__(self, index):
       
        if torch.is_tensor(index):
            index = index.tolist()
            
        image, target = self.data[index], self.targets[index]

#        image = Image.convert('RGB')
        image = Image.fromarray(image)
        
        # GRAYSCALE
#        image = image.convert('LA')
           
        image = self.transforms(image)
        
        return image, target
    
    def __len__(self):
        return len(self.data)

class FullSteelDataset(Dataset):
    
    def __init__(self, opt, dataset, train=False, valid=False, test=False):
        
        self.train = train
        self.valid = valid
        self.test = test

        if(train and not valid and not test):
            self.data = dataset['DATA']
            self.targets = dataset['LABELS']
        
        elif(valid and not train and not test):
            self.data = dataset['DATA']
            self.targets = dataset['LABELS']
        
        elif(test and not train and not valid):
            self.data = dataset['DATA']
            self.targets = dataset['LABELS']
                
        self.data = np.vstack(self.data).reshape(-1, 256, 1600, 3)
        print(self.data.shape)

        self.transforms = self._initTransforms(opt)

    def _initTransforms(self, opt):
        if(self.train):
            if(opt.augmentation==False):
                
                transforms = Transforms.Compose(
                                    [
    #                                    transforms.Resize(32, interpolation=Image.BILINEAR),
                                        Transforms.Grayscale(num_output_channels = opt.in_channels),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,),
                                                             (0.5,)),
    #                                    Transforms.Grayscale(num_output_channels = opt.in_channels)
    #                                    transforms.ToPILImage()
                                    ]
                                )
                                
            else:
                transforms = Transforms.Compose(
                                    [
                                        # AUGMENTATION
                                        Transforms.ColorJitter(brightness=0.2,
                                                                contrast=0.2, 
                                                                saturation=0.3, 
                                                                hue=0.2),
                                        
    #                                    Transforms.RandomRotation(10),
                                        Transforms.RandomAffine(degrees=10,
                                                                scale=(0.5,2),
                                                                shear=0.2),
                                        
                                        Transforms.Grayscale(num_output_channels = opt.in_channels),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,),
                                                             (0.5,))   
                                            
                                    ]
                                )                                
                                
        elif(self.valid or self.test):
            transforms = Transforms.Compose(
                                    [
                                        Transforms.Grayscale(num_output_channels = opt.in_channels),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize((0.5,),
                                                             (0.5,))  
                                    ]
                                )
        
        return transforms
    
    def __getitem__(self, index):
       
        if torch.is_tensor(index):
            index = index.tolist()
            
        image, target = self.data[index], self.targets[index]

        image = Image.fromarray(image)
           
        image = self.transforms(image)
        
#        print(type(target[0][0]))
        
        return image, target
    
    def __len__(self):
        return len(self.data)

class ImagesDataset(Dataset):
    
    def __init__(self, dataset):
        self.data = dataset['DATA']
        self.targets = dataset['LABELS']
        self.masked = dataset['MASKED']
        self.samples = dataset['SAMPLES']
        
        self.data = np.vstack(self.data).reshape(-1, 256, 1600, 3)
        print(self.data.shape)
        
    def __getitem__(self, index):

        sample, image, target = self.samples[index], self.data[index], self.targets[index]
        masked = self.masked[index]
        
        return sample, image, target, masked
    
    def __len__(self):
        return len(self.data)

class ASDataset(Dataset):
    
    def __init__(self, dataset):
        self.data = dataset['DATA']
        self.targets = dataset['LABELS']
        self.score = dataset['AS']
        self.samples = dataset['SAMPLES']
        
        self.data = np.vstack(self.data).reshape(-1, 256, 1600, 3)
        print(self.data.shape)
        
    def __getitem__(self, index):

        sample = self.samples[index] 
        image, target = self.data[index], self.targets[index]
        a_score = self.score[index]
        
        return sample, image, target, a_score
    
    def __len__(self):
        return len(self.data)
    
class TLDataset(Dataset):
    
    def __init__(self, dataset):
        self.data = dataset['DATA']
        self.targets = dataset['LABELS']
        self.masked = dataset['MASKED']
        self.samples = dataset['SAMPLES']
        
        self.data = np.vstack(self.data).reshape(-1, 256, 1600, 3)
        print(self.data.shape)
        
    def __getitem__(self, index):

        sample, image, target = self.samples[index], self.data[index], self.targets[index]
        masked = self.masked[index]
        
        return sample, image, target, masked
    
    def __len__(self):
        return len(self.data)
#%%
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

def getCifar10(opt):
    
    print('>Loading Cifar Dataset')
    
    splits = ['train', 'test']
    shuffle = {'train': True, 'test': False}
    drop_last = {'train': True, 'test': True}
    transform = transforms.Compose(
        [
    #                transforms.Resize(opt.isize),
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    classes = {
        'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
    }
    
    dataset = {}
    dataset['train'] = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset['test'] = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    #a = CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    dataset['train'].data, dataset['train'].targets, \
    dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
        trn_img=dataset['train'].data,
        trn_lbl=dataset['train'].targets,
        tst_img=dataset['test'].data,
        tst_lbl=dataset['test'].targets,
    #            abn_cls_idx=classes[opt.abnormal_class],
        abn_cls_idx=classes['car'],
    #            manualseed=opt.manualseed
    )
    
    dataset['train'].targets = dataset['train'].targets.astype('float64')
    dataset['test'].targets = dataset['test'].targets.astype('float64')
    
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batch_size,
                                                 shuffle=shuffle[x],
    #                                                     num_workers=int(opt.workers),  
                                                 drop_last=drop_last[x],
    #                                                     worker_init_fn=(None if opt.manualseed == -1
    #                                                     else lambda x: np.random.seed(opt.manualseed))
                                                )
                  for x in splits}
    
    return dataloader


def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

