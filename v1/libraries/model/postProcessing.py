#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:10:29 2019

@author: daniele
"""

#%% IMPORTS
import cv2 as cv
from libraries.model.adModel import AnomalyDetectionModel, loadModel
import numpy as np
from matplotlib import pyplot as plt

from libraries.tests import test
from libraries.dataset_package import dataset_manager as dm
from libraries.dataset_package.dataset_manager import Shape
from libraries.utils import Paths
paths= Paths()

import pandas as pd
#%%

class postProcessing():
    
    def __init__(self, adModel):
        
        self.model = adModel
        self.scores = adModel.anomaly_scores
        
        
    def getSample(self):
        
        shape = Shape(32,32)
        model_name = self.model.opt.name
        train = pd.read_csv(paths.csv_directory + 'train_unique.csv', index_col=0)
        index = 1070
        stride = 4
        
        img, _, mask = dm.extractPatchesForTest(train, index, shape, stride, model_name)
        
#%% TEST
        
