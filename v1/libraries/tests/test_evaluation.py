

#%% IMPORTS

import libraries.evaluate as ev
import numpy as np

#%%
labels = np.zeros(100)
labels

scores = np.random.randn(100)
scores

#%%
ev.roc(labels, scores)