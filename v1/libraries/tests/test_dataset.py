#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:20:04 2019

@author: daniele
"""
#%% IMPORTS

from libraries.model.dataset import generateDataloader, getCifar10, loadDataset
from libraries.model.dataset import collectAnomalySamples, collectNormalSamples
from libraries.model.options import Options
from libraries.utils import Paths
paths = Paths()
from matplotlib import pyplot as plt
import torch
#%%

def test_loadData():
    opt = Options
    
    loadDataset(opt, test='mixed')

