#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:48:23 2019

@author: daniele
"""

#%% IMPORTS
#import sys
#sys.path.append("../../")

import os
#curr_path = '/media/daniele/Data/Tesi/Practice/Code/ganomaly/ganomaly_master/dataset_package'
#os.chdir(curr_path)

import numpy as np
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy
from time import time

#libpath = '/media/daniele/Data/Tesi/Practice/Code/ganomaly/ganomaly_master/libraries'
#os.chdir(libpath)
from libraries.model.options import Options
from libraries.utils import Paths

#os.chdir(curr_path)
paths = Paths()
opt = Options()
#%% PATHS
curr_path = '/media/daniele/Data/Tesi/Practice/Code/ganomaly/ganomaly_v2/dataset_package'

base_path = '/media/daniele/Data/Tesi/Practice/'
dataset_path = base_path + '/Dataset/severstal-steel-defect-detection/'
extracted_path = base_path + 'Code/Severstal/Extracted_images/'
train_images_dir = dataset_path + 'train_images/'

my_dataset_dir = '/media/daniele/Data/Tesi/Practice/Dataset/my_dataset/'

#patched_images_dir = my_dataset_dir + 'patched_images/'
patched_images_dir = paths.patched_images_dir_50

#normal_patches_dir = my_dataset_dir + 'patches/'
#normal_patches_dir = paths.normal_images_path_50
normal_patches_dir = paths.patched_normal_center
#anomalous_patches_dir = my_dataset_dir
#anomalous_patches_dir = paths.anomalous_images_path_50
anomalous_patches_dir = paths.patched_anom_center
#anomalous_patches_dir = my_dataset_dir + 'patches/Anomalous/'

#%% CONSTANTS

MAX_SIZE_X = 1600
MAX_SIZE_Y = 256

MATPLOTLIB = 'matplotlib'
OPENCV = 'opencv'

ORIGINAL_IMAGE = 'original_image'
MASKED_IMAGE = 'masked_image'
GRAY_IMAGE = 'gray_image'

BLACK_THRESHOLD_MEDIAN_PATCH = 30
WHITE_THRESHOLD_MEDIAN_PATCH = 220

THRESHOLD_ANOMALY = opt.anom_perc

# BGR COLORS
RED         = (0,0,255)
GREEN       = (0,255,0)
BLUE        = (255,0,0)
GRAY        = (225,225,225)
LIGHT_BLUE  = (255,255,0)

COLOR_PATCH = GRAY
COLOR_CENTER = LIGHT_BLUE

#%% CLASSES

class Shape():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __str__(self):
        return '(x,y) -> ({},{})'.format(self.x, self.y)
    
    def __repr__(self):
        return '({},{})'.format(self.x, self.y)
    
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __str__(self):
        return '(x,y) -> ({},{})'.format(self.x, self.y)
    
    def __repr__(self):
        return '({},{})'.format(self.x, self.y)

class PatchClass():
    
    def __init__(self, center, shape, patchIMG, anomaly=False):
       
        self.shape=shape
        self.anomaly=anomaly
        self.center=center
        self.patch_image=patchIMG
        self.computePerimeter()
        self.model = None

    def computePerimeter(self):
#        print(self.shape.y)
        if(self.shape is not None and self.center is not None):
            y_down  = int(self.center.y - self.shape.y/2)
            y_up    = int(self.center.y + self.shape.y/2)
            x_left  = int(self.center.x - self.shape.x/2)
            x_right = int(self.center.x + self.shape.x/2)
            
            self.x_range = (x_left, x_right)
            self.y_range = (y_down, y_up)
        
    def setScore(self, score, threshold):
        self.score = score
        self.threshold = threshold

    def show(self):
        
        try:
            plt.imshow(self.patch_image)
        except:
            print('Patch not loaded')
          
    def getPatchImage(self, image):

        return image[self.y_range[0] : self.y_range[1],
                     self.x_range[0]: self.x_range[1]]        
    
    def __str__(self):
        return 'Center: ({},{}) \t Shape: ({},{})'.format(self.center.x,
                                                            self.center.y,
                                                            self.shape.x,
                                                            self.shape.y)
        

def generatePatches(data, labels):
    patches = []
    
    for i in range(0, len(data)):
        patches.append(PatchClass(data[i].center, data[i].shape, data[i].patch_image, labels[i]))
        
    return patches

class Image():
    '''
    '''
    
    def __init__(self, filename):       
        self.filename=filename
        self.patches = []
        self.loadImage(train_images_dir)
        self.normal_dir = normal_patches_dir
        self.anomalous_dir = anomalous_patches_dir
        self.folder_save = None
        self.model_name = None
        #-----check
        self.count = 0
        #------------
        
    def loadImage(self, directory):
        path = directory + self.filename
        
        original_image = cv2.imread(path)
        self.original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        self.masked_image = None
        self.gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        self.h = self.original_image.shape[0]
        self.w = self.original_image.shape[1]
        
        self.patchedImage = deepcopy(self.original_image)
        
    def initPatchedImage(self, imageFrom=GRAY_IMAGE):
        '''
        PARAMS:
            option: ORIGINAL_IMAGE -> copy of original image
                    MASKED_IMAGE   -> copy of masked image
                    GRAY_IMAGE     -> copy of gray image
        '''

        if(imageFrom == ORIGINAL_IMAGE):
            self.patchedImage = deepcopy(self.original_image)
        elif(imageFrom == MASKED_IMAGE):
            self.patchedImage = deepcopy(self.masked_image)
        elif(imageFrom == GRAY_IMAGE):
            self.patchedImage = deepcopy(self.gray_image)
        else:
            raise(ValueError('Wrong imageFrom field '))
            
        if(len(self.patchedImage.shape)==3):
            COLOR_PATCH = GRAY
            COLOR_CENTER = LIGHT_BLUE
        elif(len(self.patchedImage.shape)==2):
            COLOR_PATCH = 255
            COLOR_CENTER = 255
    
    def show(self, img=None, output=OPENCV):
        '''
        PARAMS:
            - output:   OPENCV      -> show image with opencv
                        MATPLOTLIB  -> show image with matplotlib
        '''
        title = self.filename        
        multiple = type(img) is list        
        
        if(img is None):
            img = self.original_image
        
        if(output == MATPLOTLIB):
            plt.imshow(img)
            
        elif(output == OPENCV):
            
            if(multiple):
                self.multipleImShow(title, img)
            else:
                self.imshow(title, img, multiple)
                
        elif(output == None):
            pass
        
        else:
            raise(Exception('Wrong Option provided'))
        
    def imshow(self, title, img):
        
        while(1):
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(title, img)
            
            k = cv2.waitKey(30)
            
            # ESC to stop watching
            if(k==27):
                break
        cv2.destroyAllWindows()
        
    def multipleImShow(self, title, imgList):
        
        while(1):
            i=0
            for img in imgList:
                cv2.namedWindow(title + str(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow(title + str(i), img)
                i += 1
            k = cv2.waitKey(30)
            
            # ESC to stop watching
            if(k==27):
                break
                
        cv2.destroyAllWindows()

    def _testPartition(self, patch_shape, stride, mask):
        
        self.stride = stride
        self.patches = []
        startRow = patch_shape.y//2
        startCol = patch_shape.x//2
#        print(startRow)
        rows = len(self.original_image)
        
        for row in range(startRow, rows-startRow+1, stride):
            cols = len(self.original_image[row])
            
            for col in range(startCol, cols-startCol+1, stride):
                origin = Point(col,row)
                patchIMG = getPatchImage(self.original_image, origin, patch_shape)
#               
                if(checkMedianThreshold(patchIMG)==False):
#                    print('Non-considerable patch at {}x{}'.format(origin.y, origin.x))
                    pass
                else:
                    patch = PatchClass(origin, patch_shape, patchIMG)

                    if(mask is not None):         
                        patch.anomaly = checkAnomaly(patch, mask)
                    
                    self.patches.append(patch)
                    self.drawPatch(patch, COLOR_PATCH)
        
        return self.patches
        

    def _partition(self, shape, origin, mask=None):
        '''
        PARAMS:
            - shape  : patch shape
            - origin : (x,y) coordinate starting pixel
        
        RETURNS:
            - True  -> succeded partiotion 
            - False -> failed partition
        '''        
#        print(0)
        h_p = shape.y
        w_p = shape.x
#        print(h_p)
#        print(w_p)
        
#        print('Mask, ', mask)
            
        x_left = int(origin.x - shape.x/2)
        x_right = int(origin.x + shape.x/2)
        y_down = int(origin.y - shape.y/2)
        y_up = int(origin.y + shape.y/2)
        
#        print('-Limits')
#        print(x_left, x_right, y_down, y_up, origin)
        patchImage = self.gray_image[y_down : y_up, x_left : x_right]
        
        # CHECKS
        if(origin.y + h_p/2 > self.h or origin.y - h_p/2 < 0 or origin.x + w_p/2 > self.w or origin.x + w_p/2 < 0):
            raise(Exception('Dimension does not fit the image'))
            
        elif(checkMedianThreshold(patchImage) == False):
#            print('Out of Median Threshold -- Patch Discarded')
#            print('Patch discarded')
            return False
        
        # SUCCESSFULLY PARTITION
        patch = PatchClass(origin, shape, patchImage)
        
        if(mask is not None):         
            patch.anomaly = checkAnomaly(patch, mask)
#            print(patch.anomaly)
            
        self.patches.append(patch)
        
        self._drawPartition(patch)            
        
        if(patch.anomaly == True):
            return False
        else:
            return True
        
    def _drawPartition(self, patch, output=None, save=True):
        
        # ----------SAVE MODE------------
        if(save):
            os.chdir(self.normal_dir)
#            print(patch.anomaly)
            if(patch.anomaly == True):
                os.chdir(self.anomalous_dir)
        
            cv2.imwrite(self.filename + '_' + str(patch.center), patch.patch_image)      
            
            os.chdir(curr_path)              
        #-------------------------------
        
        # NORMAL PATCH DRAWN ON IMAGE
        if(patch.anomaly == False):
            vertex1 = (patch.x_range[0], patch.y_range[1])
            vertex2 = (patch.x_range[1], patch.y_range[0])

            cv2.rectangle(self.patchedImage, vertex1, vertex2, color=COLOR_PATCH, thickness=1)

            self.patchedImage[patch.center.y, patch.center.x] = 255
            
    def drawPatch(self, patch, color):
        vertex1 = (patch.x_range[0], patch.y_range[1])
        vertex2 = (patch.x_range[1], patch.y_range[0])
    
        cv2.rectangle(self.patchedImage, vertex1, vertex2, color=COLOR_PATCH, thickness=1)
    
        self.patchedImage[patch.center.y, patch.center.x] = 255
    
    def drawAnomalies(self, method='simple', patches=None, save=True):
        
        return
        
        
    
    def drawAnomaliesSimple(self, patches=None, save=True, info=''):
        '''
        Description:
            It draws anomalies as they are predicted by the model
        '''
        mask = np.zeros([256, 1600]) -1
        count = 0
        
        if(patches is None):
            patches = self.patches
        else:
            patches = patches
        
        for patch in patches:
            
#            print(patch.anomaly)
            if(patch.anomaly == True):
                count += 1
                y1, y2 = int(patch.y_range[0]), int(patch.y_range[1])
                x1, x2 = int(patch.x_range[0]), int(patch.x_range[1])
                        
                mask[y1:y2, x1:x2] = 1
#                print('Mask len: {}'.format(len(np.where(mask[0]))))
#                break
        self.patchedImage[mask==1, 2] = 150
        img_masked = deepcopy(self.masked_image)
        img_masked[mask==1, 2] = 255
        
        if(save):
            directory = self.folder_save + '/'            
            os.chdir(directory)
            print('\n')
            print('> Saving to ..{}\n'.format(directory))
#            cv2.imwrite('3'+ '_{}_(Simple)ANOMALIES_{}'.format(self.model_name, self.filename), self.patchedImage)
            cv2.imwrite('3'+ '_{}_(Simple)MY_ANOMALY_MASK_stride:{}_{}_{}'.format(self.model_name, self.stride,
                                                                        self.filename, 
                                                                        info), img_masked)
            os.chdir(curr_path)
            
        mask[mask==-1]=0    
        
        return mask, count

    def drawAnomaliesMajVoting(self, patches=None, save=True, info=''):
        '''
            Description:
                It draws anomalies taking into account the multiple outcome assigned 
                to a pixel
        '''
        x_size = 1600
        y_size = 256
        maxVotes = 64
        
        mask = np.zeros([maxVotes, y_size, x_size])-1
#        votes_dict = createVotesDict(maxVotes)
        
#        count = 0
#        nVotes = []
        start = time()
        print('--> Majority Voting method')
        
        
        if(patches is None):
            patches = self.patches
        else:
            patches = patches
        
        for patch in patches:
#                shape = patch.shape
#                score = patch.score
            
            y1, y2 = int(patch.y_range[0]), int(patch.y_range[1])
            x1, x2 = int(patch.x_range[0]), int(patch.x_range[1])
#                print(y1)
#                print(y2)
            for x in range(x1,x2):
                for y in range(y1,y2):    
#                        print(y)
                    indexVote = findVoteIndex(mask, x, y)
                    
                    if(patch.anomaly):
                        mask[indexVote, y, x] = 1
                    else:
                        mask[indexVote, y, x] = 0
        
        final_mask = np.zeros([y_size, x_size])
        
        for x in range(x_size):
            for y in range(y_size):
                value = computeVoting(mask, x, y)
                
                final_mask[y,x] = value
        
        img_masked = deepcopy(self.masked_image)
        img_masked[final_mask==1, 2] = 255
         
        end = time()
        
        if(save):
#            folder_name = self.filename.split('.')[0]
            directory = self.folder_save
            os.chdir(directory)
            print('\n')
            print('> Saving to ..{}\n'.format(directory))
#            cv2.imwrite('4'+ '_(Smart)ANOMALIES_{}'.format(self.filename), self.patchedImage)
            cv2.imwrite('5'+ '_{}_(Maj-Voting)MY_ANOMALY_MASK_stride:{}_{}_{}'.format(self.model_name,
                                                                        self.stride,
                                                                        self.filename,
                                                                        info), img_masked)
            os.chdir(curr_path)
#
        spent_time = end-start
        minutes = spent_time // 60
        seconds = ((spent_time / 60) - minutes) * 60
        print('> Time spent: {:.0f} min {:.0f} sec'.format(minutes, seconds))   
        
        return final_mask
    
    def drawAnomaliesThesholding(self, save=True):
        x_size = 1600
        y_size = 256
        maxVotes = 500
        
        map_size = (128,128)
        
        mask = np.zeros([maxVotes, y_size, x_size])-1
#        votes_dict = createVotesDict(maxVotes)
        
#        count = 0
#        nVotes = []
        print('--> Thresholding method')
        print('> Getting scores')
        start = time()
        
        for patch in self.patches:
            score = patch.score
            
            y1, y2 = int(patch.y_range[0]), int(patch.y_range[1])
            x1, x2 = int(patch.x_range[0]), int(patch.x_range[1])
#                print(y1)
#                print(y2)
            for x in range(x1,x2):
                for y in range(y1,y2):    
#                        print(y)
                    indexVote = findVoteIndex(mask, x, y)
                    
                    mask[indexVote, y, x] = score
                    
        final_mask = np.zeros([y_size, x_size])
        
        print('> Getting average scores')
        for x in range(x_size):
            for y in range(y_size):
                value = computeThresholding(mask, x, y)
                
                final_mask[y,x] = value
        
        img_masked = deepcopy(self.masked_image)
        print('> Getting anomaly mask')
        img_masked = maskingByMaps(map_size, final_mask, img_masked)
#        return maskingByMaps(map_size, final_mask, img_masked)
        
        end = time()
        
        if(save):
#            folder_name = self.filename.split('.')[0]
            directory = self.folder_save   
            os.chdir(directory)
            print('\n')
            print('> Saving to ..{}\n'.format(directory))
#            cv2.imwrite('4'+ '_(Smart)ANOMALIES_{}'.format(self.filename), self.patchedImage)
            cv2.imwrite('5'+ '_{}_(Thr.Map)MY_ANOMALY_MASK_stride:{}_{}'.format(self.model_name, self.stride,
                                                                        self.filename), img_masked)
            os.chdir(curr_path)
#       
        spent_time = end-start
        minutes = spent_time // 60
        seconds = ((spent_time / 60) - minutes) * 60
        print('> Time spent: {:.0f} min {:.0f} sec'.format(minutes, seconds))    
        
        return final_mask
        
    
    def drawAnomaliesThresholdindAll(self, save=True):
        '''
        Description:
            It draws anomalies computing the threshold of all scores
        
        '''
        x_size = 1600
        y_size = 256
        mask = np.zeros([y_size, x_size]) -1
        count = 0
        for patch in self.patches:
            
            if(patch.anomaly == True):
                count += 1
                y1, y2 = int(patch.y_range[0]), int(patch.y_range[1])
                x1, x2 = int(patch.x_range[0]), int(patch.x_range[1])
                
                score = patch.score
#                for x in range(x1, x2):
#                    for y in range(y1, y2):
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if(mask[y,x] == -1):
                            mask[y,x] = score
                        else:
                            mask[y,x] = (mask[y,x] + score)/2  
        
        mask[mask==-1] = 0
        
#        model_threshold = patch.threshold
#        print('--> Model threshold: {}'.format(model_threshold))
        
        avg_threshold = np.sum(mask[mask!=-1]) / (x_size*y_size)
#        print(avg_threshold)
        avg_threshold = np.sum(mask[mask!=-1]) / (np.count_nonzero(mask))        
#        print('--> Averaging threshold: {}'.format(avg_threshold))
        
#        self.patchedImage[mask > threshold, 2] = 150
#        img_masked_model = deepcopy(self.masked_image)
#        img_masked_model[mask > model_threshold, 2] = 255
        
        img_masked_avg = deepcopy(self.masked_image)
        img_masked_avg[mask > avg_threshold, 2] = 255
        
        if(save):
#            folder_name = self.filename.split('.')[0]
            directory =self.folder_save + '/'            
            os.chdir(directory)
            print('\n')
            print('> Saving to ..{}\n'.format(directory))
#            cv2.imwrite('4'+ '_(Smart)ANOMALIES_{}'.format(self.filename), self.patchedImage)
#            cv2.imwrite('3'+ '_(Smart-model_th)MY_ANOMALY_MASK_stride:{}_{}'.format(self.stride,
#                                                                        self.filename), img_masked_model)
            cv2.imwrite('4'+ '_{}_(Thr.All)MY_ANOMALY_MASK_stride:{}_{}'.format(self.model_name, self.stride,
                                                                        self.filename), img_masked_avg)
            os.chdir(curr_path)
            
#        mask[mask > avg_threshold] = 1
        
        return mask, count
#%% FUNCTIONS
        
#def createVotesDict(maxVotes):
#    vote_dict = {}
#    
#    for i in range(maxVotes):
#        vote_dict.update({str(i):0})
#    
#    return vote_dict
#   
def maskingByMaps(maps, mask, img):
    map_size = maps
    img_size = img.shape
#    print(map_size)
#    print(img_size)
    step_x = img_size[1] // map_size[0]
    step_y = img_size[0] // map_size[1]
    
#    print(step_x)
#    print(step_y)
    
    for x in range(0, img_size[1]-step_x, step_x):
        for y in range(0, img_size[0]-step_y, step_y):
            x_left = x
            x_right = x + step_x
            y_upper = y
            y_lower = y + step_y
            
            submask = mask[y_upper:y_lower, x_left:x_right]
            sub_img = img[y_upper:y_lower, x_left:x_right]
            
#            thr = np.sum(mask[y_upper:y_lower, x_left:x_right]) / (submask.shape[0]*submask.shape[1])
            thr = np.average(submask)
            
#            print('------> threshold')
#            print(thr)
#            print('------> threshold.avg numpy')
#            print(thr_avg)
#            print('----> mask shape')
#            print(submask.shape)
            
            sub_img[submask > thr, 2] = 255
            plt.plot(sub_img)
            plt.show()
#            print('> x: ', x)
#            print('> y: ', y)
#    return [submask, thr]        
    return img
            
def computeThresholding(mask, x, y):
    votes = []
    
    for i in range(len(mask)):
        value = mask[i, y, x]
        if(value != -1):
            votes.append(mask[i, y, x])
    
    # All -1 case, where there are no patch
    if(len(votes) == 0):
        return 0
    else:
        return np.average(votes)
    

def computeVoting(mask, x,y):
    votes = []
    
    for i in range(len(mask)):
        value = mask[i, y, x]
        if(value != -1):
            votes.append(mask[i, y, x])
    
#    print('Votes: {}'.format(votes))
    if(np.sum(votes) > (len(votes)/2)):
        return 1
    else:
        return 0
    
def findVoteIndex(mask, x, y):
    '''
    Description:
        It finds the first unused index of the vote index array
    '''
    
    for voteIndex in range(len(mask)):
        if(mask[voteIndex][y][x] == -1):
            return voteIndex
        
    raise Exception('Vote index not found')
    
def findFirstMaskElement(x_size, y_size, mask):
    for x in range(x_size):
        for y in range(y_size):
            if(mask[y,x] != -1):
                print('center: {}x{}'.format(x,y))
                return x,y

def getPatchImage(image, origin, shape):

    x_left = int(origin.x - shape.x/2)
    x_right = int(origin.x + shape.x/2)
    y_down = int(origin.y - shape.y/2)
    y_up = int(origin.y + shape.y/2)
    
    return image[y_down : y_up, x_left : x_right]

def drawPatch(img, patch, color):
    vertex1 = (patch.x_range[0], patch.y_range[1])
    vertex2 = (patch.x_range[1], patch.y_range[0])

    cv2.rectangle(img.patchedImage, vertex1, vertex2, color=COLOR_PATCH, thickness=1)

    img.patchedImage[patch.center.y, patch.center.x] = 255

def extractPatchesOptimized(train, start, end, nPatches, shape):
    count = start
    
    for row in train.index[start : end]:
#    print(row)
        filename    = train.iloc[row].Image_Id
        enc_pixels  = train.iloc[row].Encoded_Pixels
        #    print(filename)
        #    print(enc_pixels)   threshold
        img = Image(filename)
        #    img = Image('002fc4e19.jpg')
        
        mask = computeMask(enc_pixels, img)    
        img = applyMask(img, mask)
        
        os.mkdir(normal_patches_dir + str(count) + '/')
        os.mkdir(anomalous_patches_dir + str(count) + '/')
        
        print('Salvataggio No ', count)
        
        img.normal_dir = normal_patches_dir + str(count) + '/'
        img.anomalous_dir = anomalous_patches_dir + str(count) + '/'
        
        _getPatchedImage(img, nPatches, shape, mask, output=None)
#       partition 
        
        os.chdir(patched_images_dir)
        cv2.imwrite(str(count) + '.PATCHED_{}x{}_'.format(shape.x, shape.y) + img.filename, img.patchedImage)
        cv2.imwrite(str(count) + '_MASK_' + img.filename , img.masked_image)
        os.chdir(curr_path)

        count += 1
#        print(count)
            
 
def extractPatchesForTest(train, index, shape, stride, model_name):

    filename    = train.iloc[index].Image_Id
    enc_pixels  = train.iloc[index].Encoded_Pixels
    
    img = Image(filename)
    
    mask = computeMask(enc_pixels, img)            
    img = applyMask(img, mask)
    img.model_name = model_name
    
    father_folder = paths.test_patched + model_name + '/'
    if(os.path.exists(father_folder) == False):
        print('Creating directory ..{}'.format(father_folder))
        os.mkdir(father_folder)
    
    img.folder_save = father_folder + str(index) + '_' + filename.split('.')[0] + '/'
#    directory = paths.test_patched + img.folder_save + '/'
    
    if(os.path.exists(img.folder_save) == False):
        print('Creating directory ..{}'.format(img.folder_save))
        os.mkdir(img.folder_save)
    
    
    print('Salvataggio')
    
    patches = img._testPartition(shape, stride, mask)
    
    os.chdir(img.folder_save)
    print('> Saving to ..{}'.format(img.folder_save))
    cv2.imwrite('0'+ '_{}_{}_{}x{}'.format(model_name, img.filename, shape.x, shape.y), img.original_image)
    cv2.imwrite('1'+ '_{}_PATCHED_{}x{}_stride:{}_'.format(model_name, shape.x, shape.y, stride) + img.filename, img.patchedImage)
    cv2.imwrite('2' + '_MASK_' + img.filename , img.masked_image)
#    cv2.imwrite('3')
    os.chdir(curr_path)

    return img, patches, mask

def _getPatchedImage(img, nPatches, shape, mask, output=None):
    '''
    PARAMS:
        - img       : image instance from Image
        - nPatches  : n patches to be taken from img
        - shape     : patch shape
        - output    : where to show patched image (MATPLOTLIB / OPENCV)
        - save      : bool for saving or not (TRUE / FALSE)
    '''
#    print('_GetPatchedImage')
    points = []

    x_ll = shape.x
    x_ul = MAX_SIZE_X - shape.x
    y_ll = shape.y
    y_ul = MAX_SIZE_Y - shape.y
    
    # INIT WHERE TO DRAW PATCH
    img.initPatchedImage()
    
#    print('\nPoints')
    
    for i in range(0, nPatches):
        
        point = randomPoint(x_ll, x_ul, y_ll, y_ul)
#        point = Shape(50,50)
        # CHECK WHETHER THE POINT IS ALREADY TAKEN
        while(checkExistency(points, point) == False):
            print('Discarded same point already taken')
            point = randomPoint(x_ll, x_ul, y_ll, y_ul)    
 
        points.append(point)

        # CHECK IF PARTITION IS GOOD OR NOTX
        while(img._partition(shape, point, mask) == False):
#            print(img.masked_image)
#            print('\tDiscarded point: ({},{})'.format(point.x, point.y))
#            imshow('Find discarded pixel', img.masked_image)
            point = randomPoint(x_ll, x_ul, y_ll, y_ul)
        
#            # SUCCESFULLY PARTITION
#            patch = img.patches[-1]
##            print('provaaaaaa ', patch.anomaly)
#            img._drawPartition()
#        print(i)


def getPatchedImage(img, nPatches, shape, mask, output=None):
    '''
    PARAMS:
        - img       : image instance from Image
        - nPatches  : n patches to be taken from img
        - shape     : patch shape
        - output    : where to show patched image (MATPLOTLIB / OPENCV)
        - save      : bool for saving or not (TRUE / FALSE)
    '''

    points = []

    x_ll = shape.x
    x_ul = MAX_SIZE_X - shape.x
    y_ll = shape.y
    y_ul = MAX_SIZE_Y - shape.y
    
    # INIT WHERE TO DRAW PATCH
    img.initPatchedImage()
    
    print('\nPoints')
    
    for i in range(0, nPatches):
        
        point = randomPoint(x_ll, x_ul, y_ll, y_ul)
        
        # CHECK WHETHER THE POINT IS ALREADY TAKEN
        while(checkExistency(points, point) == False):
            print('Discarded same point already taken')
            point = randomPoint(x_ll, x_ul, y_ll, y_ul)    
 
        points.append(point)

        # CHECK IF PARTITION IS GOOD OR NOT
        while(img.partition(shape, point, mask) == False):
#            print(img.masked_image)
            print('\tDiscarded point: ({},{})'.format(point.x, point.y))
#            imshow('Find discarded pixel', img.masked_image)
            point = randomPoint(x_ll, x_ul, y_ll, y_ul)
        
#        # SUCCESFULLY PARTITION
#        patch = img.patches[-1]

def extractImages(train, start, end):
    count = start
    
    for row in train.index[start : end]:
        filename    = train.iloc[row].Image_Id
        enc_pixels  = train.iloc[row].Encoded_Pixels
        img = Image(filename)
        
        mask = computeMask(enc_pixels, img)    
        img = applyMask(img, mask)
        
        print('Salvataggio No ', count)
#       partition 
        
        os.chdir('/media/daniele/Data/Tesi/Practice/Dataset/my_dataset/test_images/labeled_images')
        
        cv2.imwrite(str(count) + '.Original_Image_' + img.filename,
                                    img.original_image)
        cv2.imwrite(str(count) + '_MASK_' + img.filename , img.masked_image)
        os.chdir(curr_path)
        
        count += 1
        
def checkAnomaly(patch, mask):
    '''
    DESCRIPTION:
        It checks the anomaly relative to ONLY the patch center
    
    PARAMS:
        - patch : patch to be checked if it is anomalous
        - mask  : mask that shows the anomaly of entire image
    
    RETURNS:
        - True  : the patch is anomaluos
        - False : the patch is notanomaluos
    '''
    
    center = patch.center
    
    check = mask[center.y, center.x]
    
    if(check):
        return True
    else:
        return False
         
def checkExistency(points, point):
    
    for item in points:
        # ALREADY EXISTS
        if(item.x == point.x and item.y == point.y):
            return False
        
        # NEW VALUE
        else:
            return True
    
def checkMedianThreshold(patchImage):
#    print('---Check')
#    print(patchImage)
    median = np.median(patchImage)
#    print('-Median: ', median)
    
    if(median > BLACK_THRESHOLD_MEDIAN_PATCH and median < WHITE_THRESHOLD_MEDIAN_PATCH):
        return True
    else:
#        print('-Median: ', median)
        return False
    
    
def computeMask(enc_pixel, img):
    
    width = img.h
    height= img.w
    
    mask= np.zeros( width*height ).astype(np.uint8)
    if(enc_pixel == 0):
#        print('Zeroooooo')
        return np.zeros((width, height))
    
    array = np.asarray([int(x) for x in enc_pixel.split()])
    #print(array)
    starts = array[0::2]
#    print(starts)
    lengths = array[1::2]
#    print(lengths)
    
    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    mask = np.flipud(np.rot90( mask.reshape(height,width), k=1))
#    print(final)
    
    return mask
        
def applyMask(img, mask):
    
    img.masked_image = deepcopy(img.original_image)    
    img.masked_image[mask==1, 0] = 255
    
    return img
def randomPoint(x_ll, x_ul, y_ll, y_ul):
    '''
    
    PARAMS:
        - x_ll : x lower limit
        - x_ul : x upper limit
        - y_ll : y lower limit
        - y_up : y upper limit
    
    '''
    x = np.random.randint(x_ll, x_ul)
    y = np.random.randint(y_ll, y_ul)
    
    point = Shape(x,y)
    
    return point
