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

#%%

class postProcessing(self):
    
    def __init__(self, adModel):
        
        self.scores = adModel.anomaly_scores