# -*- coding: utf-8 -*-

#%%
import numpy as np
from matplotlib import pyplot as plt
from time import time
from libraries.utils import timeSpent
#%%

def anomalyScoreFromDataset(model, dataset, stride, patch_size):
    
    anomalyScoreMap = []
    gtMap = []
    
    start = time()
    
    for image, mask in dataset:
        
        anom_score, gt = anomalyScoreFromImage(model, image, mask,
                                               stride, patch_size)
        
        anomalyScoreMap.append(anom_score)
        gtMap.append(gt)
        
    end = time()
    
    timeSpent(end-start)
    
    return anomalyScoreMap, gtMap

def anomalyScoreFromImage(model, image, mask, stride, patch_size):
    h = image.shape[0]
    w = image.shape[1]
    
    bound_x = (w - patch_size)//stride
    bound_y = (h - patch_size)//stride
    
    x, y = patch_size//2, patch_size//2
    i, j = 0, 0
    
    counter = 0
    
    anomalyMap = np.zeros([bound_y, bound_x]) - 1
    maskMap = np.zeros([bound_y, bound_x]) - 1
#    print(anomalyMap.shape)
    
    patch = nextPatch(image, x, y, stride, patch_size)
    center_x = patch[1]
    center_y = patch[2]

    
    while(patch is not None):
        counter += 1
#        print(counter)
        
        score = anomalyScore(model, patch[0])
#        print(score)

        anomalyMap[j,i] = score
#        print(center_x)
#        print(center_y)
        maskMap[j,i] = mask[center_y, center_x]
        
        if(i < bound_x-1):
            i += 1
            x = x + stride
            
        elif(j < bound_y-1):
            j += 1
            i = 0
            y = y + stride
            x = patch_size//2
            
        else:
            return anomalyMap, maskMap
        
        patch = nextPatch(image, x, y, stride, patch_size)
        center_x = patch[1]
        center_y = patch[2]
    
def nextPatch(image, x, y, stride, patch_size):
    
    h = image.shape[0]
    w = image.shape[1]
    
    assert x <= w-patch_size/2 and y <= h-patch_size/2, 'Patch dimension out of image'
    
    if(x + patch_size/2 + stride <= w):
        y = y
        x = x + stride
        
    elif(y + patch_size/2 + stride <=h):
        x = patch_size
        y = y + stride
    
    else:
        return None
        
    patch = image[y - patch_size//2 : y + patch_size//2, x - patch_size//2 : x + patch_size//2]
    
    return [patch, x, y]

def anomalyScore(model, patch):
    prediction, score, _ = model.predict(patch)
    
    return score