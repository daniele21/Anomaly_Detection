

#%% IMPORTS

import libraries.model.evaluate as ev
import numpy as np

#%%
labels = np.zeros(100)
labels

scores = np.random.randn(100)
scores

#%%
ev.roc(labels, scores)

#%%
mask = np.array([[0,0,0],[0,1,1],[0,1,1]])
mask

label = np.array([[0,0,1],[0,1,1],[0,1,0]])
label

inter, union, iou = ev.IoU(mask, label)
print(iou)
print(inter)
print(union)
