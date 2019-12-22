# -*- coding: utf-8 -*-

#%%
from libraries.model import evaluate as ev
import numpy as np
from sklearn.metrics import classification_report
#%%
mask = np.array([[0,1,1],
                 [0,1,1],
                 [1,1,1]])

true_mask = np.array([[0,1,1],
                      [0,1,1],
                      [0,1,1]])

ev.IoU(mask, true_mask)

#%%
result = classification_report(mask, true_mask, output_dict=True)
result