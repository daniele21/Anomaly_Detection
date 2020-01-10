# -*- coding: utf-8 -*-

import torch

from libraries.model.options import Options
from libraries.model.classifierFCONV import ClassifierFCONV
from libraries.model.classifierFCONN import ClassifierFCONN
from libraries.torchsummary import summary


#%%
opt = Options()
model = ClassifierFCONV(opt)

summary(model.cuda(), (3,32,32))

#%%
model = ClassifierFCONN(opt)

summary(model.cuda(), (3,32,32))
