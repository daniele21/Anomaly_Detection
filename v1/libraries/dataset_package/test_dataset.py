#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 09:43:12 2019

@author: daniele
"""
#%% GET PATCHES
#---------------------------------
train = pd.read_csv('train_unique.csv', index_col=0)
start = 0
end = 1000
nPatches = 2000
shape = Shape(32,32)

dm.extractPatchesOptimized(train, start, end, nPatches, shape)
#-----------------------------------
#%%
import os
#curr_path = '/media/daniele/Data/Tesi/Practice/Code/ganomaly/ganomaly_master/dataset_package'
#os.chdir(curr_path)
#%% IMPORTS
import dataset_manager as dm
from dataset_manager import PatchClass, Image, Shape
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
#import libraries.model_options
#%% CONSTANTS

MAX_SIZE_X = 1600
MAX_SIZE_Y = 256

MATPLOTLIB = 'matplotlib'
OPENCV = 'opencv'

ORIGINAL_IMAGE = 'original_image'
MASKED_IMAGE = 'masked_image'
GRAY_IMAGE = 'gray_image'

# PATHS
base_path = '/media/daniele/Data/Tesi/Practice/'
dataset_path = base_path + '/Dataset/severstal-steel-defect-detection/'
extracted_path = base_path + 'Code/Severstal/Extracted_images/'
train_images_dir = dataset_path + 'train_images/'

patches_dir = '/media/daniele/Data/Tesi/Practice/Code/Severstal/Extracted_images/Patches/Prova'
shape = Shape(32,32)

#%% GET PATCHES
#---------------------------------
train = pd.read_csv('train_unique.csv', index_col=0)
start = 0
end = 400
nPatches = 2000
shape = Shape(32,32)

dm.extractPatchesOptimized(train, start, end, nPatches, shape)
#-----------------------------------
#%% GET PATCHES
#---------------------------------
train = pd.read_csv('train_unique.csv', index_col=0)
start = 400
end = 800
nPatches = 2000
shape = Shape(32,32)

dm.extractPatchesOptimized(train, start, end, nPatches, shape)
#-----------------------------------
#%% GET PATCHES
#---------------------------------
train = pd.read_csv('train_unique.csv', index_col=0)
start = 800
end = 1600
nPatches = 2000
shape = Shape(32,32)

dm.extractPatchesOptimized(train, start, end, nPatches, shape)
#-----------------------------------
#%% GET PATCHES
#---------------------------------
train = pd.read_csv('train_unique.csv', index_col=0)
start = 1600
end = 2000
nPatches = 2000
shape = Shape(32,32)

dm.extractPatchesOptimized(train, start, end, nPatches, shape)
#-----------------------------------
#%% GET PATCHES
#---------------------------------
train = pd.read_csv('train_unique.csv', index_col=0)
start = 0
end = 1000
nPatches = 2000
shape = Shape(32,32)

dm.extractPatchesOptimized(train, start, end, nPatches, shape)
#-----------------------------------
#%% GET PATCHES FOR TEST
train = pd.read_csv('train_unique.csv', index_col=0)
start = 1000
end = 1022
shape = Shape(32,32)
eachRow = 4
eachCol = 4

dm.extractPatchesForTest(train, start, end, shape, eachRow, eachCol)
#%% GET TEST IMAGES
train = pd.read_csv('train_unique.csv', index_col=0)
start = 1000
end = 1500
dm.extractImages(train, start, end)
#%% ANOMALY TEST
mask = np.array([[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]])
patch = PatchClass(dm.Point(3,3), Shape(32,32), None)

dm.checkAnomaly(patch, mask)
#%%
filename = train.iloc[0].Image_Id
img = Image(filename)

image = img.original_image

plt.imshow(img.gray_image)
plt.show()

plt.imshow(img.original_image)
plt.show()

#%%
a = img._testPartition(shape, 4, 4)
plt.imshow(img.patchedImage)
#a[0].show()
#%% GET PATCHES FOR TRAINING
#---------------------------------
train = pd.read_csv('train_unique.csv', index_col=0)
start = 0
end = 1000
nPatches = 2000
shape = Shape(32,32)

dm.extractPatchesOptimized(train, start, end, nPatches, shape)
#-----------------------------------

#%%
#shape = (32,32)
shape = Shape(32,32)
center = Shape(50,50)

j=0
i = 500

count=0
nPatches = 10000
save=True

# EXTRACTION IMAGES AND PATCHES
for row in train.index[j:i]:
#    print(row)
    filename    = train.iloc[row].Image_Id
    enc_pixels  = train.iloc[row].Encoded_Pixels
#    print(filename)
#    print(enc_pixels)   
    img = Image(filename)
#    img = Image('002fc4e19.jpg')
    
    mask = dm.computeMask(enc_pixels, img)    
    img = dm.applyMask(img, mask)
    
#    dm.getPatchedImage(img, 10, shape, mask, output=OPENCV, save=True)
    dm.getPatchedImage(img, nPatches, shape, mask, output=None)
    
    # Output on OPENCV
#    img.drawPartition(baseImage=GRAY_IMAGE, output=OPENCV, save=save)
    # NO OUTPUT
    img.drawPartition(baseImage=GRAY_IMAGE, output=None, save=save)
    
    count += 1
    print(count)
    

#%% LOAD IMAGE
def testImage(img, pathDir):
    img.loadImage(pathDir)
    return img.show()
#    img.partition((128,128), (100,100))
#    a = img.showPartitions()

#%% PARTITION
    
def partition(img, shape, center):
    img.partition(shape, center)
    
    img.patches[0].show()
#plt.imshow(img.original_image)
    
#%% IMSHOW
def imshow(title, img):    
    
    while(1):
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, img)
        
        k = cv2.waitKey(30)
        
        # ESC to stop watching
        if(k==27):
            break
        
    cv2.destroyAllWindows()

#%% POINTS GENERATION

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

def getPatchedImage(img, nPatches, shape, mask, output=None, save=False):

    x_ll = shape.x
    x_ul = MAX_SIZE_X - shape.x
    y_ll = shape.y
    y_ul = MAX_SIZE_Y - shape.y
    
    print('\nPoints')
    
    for i in range(0, nPatches):
        point = randomPoint(x_ll, x_ul, y_ll, y_ul)
        print('{}\t->\t{}'.format(i+1, point))

        # CHECK IF PARTITION IS GOOD OR NOT
        while(img.partition(img.masked_image, shape, point, mask) == False):
#            print(img.masked_image)
            print('Discarded point: ', point.x, point.y)
#            imshow('Find discarded pixel', img.masked_image)
            point = randomPoint(x_ll, x_ul, y_ll, y_ul)
        
    img.drawPartition(output=output, save=save)

#%%
#testImage(img, train_images_dir)
##%%
#testImage(img, train_images_dir)
#partition(img, shape, center)
#partition(img, shape, Shape(100,100))
#
#patches = img.patches
#
#img.drawPartition()

#%%
for row in train.index[0:1]:
#    print(row)
    filename    = train.iloc[row].Image_Id
    enc_pixels  = train.iloc[row].Encoded_Pixels
#    print(filename)
#    print(enc_pixels)   
    img = Image(filename)
#    img = Image('002fc4e19.jpg')
    
    mask = dm.computeMask(enc_pixels, img)    
    img = dm.applyMask(img, mask)
    
#    dm.getPatchedImage(img, 10, shape, mask, output=OPENCV, save=True)
    dm.getPatchedImage(img, 50, shape, mask, output=None, save=True)
    
    img.drawPartition(baseImage=GRAY_IMAGE, output=OPENCV, save=True)

#%%
#points = []
#
#x_ll = shape.x
#x_ul = 1600 - shape.x
#y_ll = shape.y
#y_ul = 256 - shape.y
#
#for row in train.index[0:10]:
##    print(row)
#    filename    = train.iloc[row].Image_Id
#    enc_pixels  = train.iloc[row].Encoded_Pixels
##    print(filename)
##    print(enc_pixels)   
#    img = Image(filename)
#    
#    mask = dm.computeMask(enc_pixels, img)    
#    img = dm.applyMask(img, mask)
#
#
#print('\nPoints')
#for i in range(0,10):
#    point = randomPoint(x_ll, x_ul, y_ll, y_ul)
#    print('{}\t->\t{}'.format(i+1, point))
#    
##    while(img.partition(img.masked_image, shape, point) == False):
##        imshow('Error', img.patchedImage)
##        point = randomPoint(x_ll, x_ul, y_ll, y_ul)
##        print('{}\t->\t{}'.format(i+1, point))
#    
#img.drawPartition()
#    
#    
#    
#%% Remove Duplicated images
train = pd.read_csv('train_db.csv', index_col=0)
train = train.reset_index()
a = train.Image_Id.duplicated()   
a 
b = np.where(a)
b   
train['Duplicated'] = a
#%%
def concatEncPixels(train, i, image_id):
    row = train.iloc[i]
    row_prev = train.iloc[i-1]
    
    if(row_prev.Image_Id == image_id):
        print('Same at ', i)
#        print(row_prev['Encoded_Pixels'])
        newEnc = row_prev['Encoded_Pixels'] + ' ' + row.Encoded_Pixels
        newClass= str(row_prev.Class_Id) + ' ' + str(row.Class_Id)
        train.loc[i-1, 'Encoded_Pixels'] = newEnc
        train.loc[i-1, 'Class_Id'] = newClass
#        print(row_prev.Class_Id)
        concatEncPixels(train, i-1, image_id)
        
    else:
        return

for i in range(0, len(train)-1):
    print(i)
    row = train.iloc[i]
    row_next = train.iloc[i+1]
#    print(row['duplicated'])
    if(row['Duplicated'] == True and row_next['Duplicated'] != True):
        image_id = row.Image_Id
#        print(image_id)
        concatEncPixels(train, i, image_id)
#        break

#%%

for i in b:
    train.drop(train[train['Duplicated'] == True].index, inplace=True)
    
#    break
    
#%%
train = train.reset_index()    
train = train.drop(columns='level_0')    
train.to_csv('train_unique.csv')
    
    
    
    
    
    
    
    
    
