#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:54:45 2019

@author: daniele
"""
#%%
import numpy as np

#%%

mask = np.zeros([10,10])-1
mask


a = 0.5

for x in range(0,5):
    for y in range(0,5):
        if(mask[y,x] == -1):
            mask[y,x] = a

mask
#%%
b = 0.9

for x in range(3,8):
    for y in range(0,5):
        if(mask[y,x] == -1):
            mask[y,x] = b
        else:
            mask[y,x] = (mask[y,x] + b)/2
            
mask
#%%
c = 0.2

for x in range(0,5):
    for y in range(3,8):
        if(mask[y,x] == -1):
            mask[y,x] = c
        else:
            mask[y,x] = (mask[y,x] + c)/2
            
mask
#%%
d = 0.7

for x in range(3,8):
    for y in range(3,8):
        if(mask[y,x] == -1):
            mask[y,x] = d
        else:
            mask[y,x] = (mask[y,x] + d)/2
            
mask

#%%
#mask = [[0,1,2], np.zeros([10,10]), np.ones([10,10])]
mask = np.zeros([2,5,5])
mask

np.insert(mask, axis=0, )

mask.shape
mask[0].fill(0)
mask[1].fill(1)
mask[2].fill(2)
mask
findVoteIndex(mask, 5, 5)
#%%
def findVoteIndex(mask, x, y):
    for voteIndex in range(len(mask)):
        if(mask[voteIndex][y][x] == -1):
            return voteIndex
