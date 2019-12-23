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

#%%

a = np.zeros((3,3))
a[0][2] = 1
print(a)

b = np.ones((3,3))
print(b)

np.bitwise_and(a,b)
#a & b

#%%
c = np.zeros((3,3))
c[2,2] = 1
print(c.shape)
c = c.astype(int)
c

c&c
