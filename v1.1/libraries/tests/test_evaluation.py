

#%% IMPORTS

import libraries.model.evaluate as ev
import seaborn as sn
import numpy as np
from scipy.stats import cumfreq
#%%
labels = np.zeros(100)
labels

scores = np.random.randn(100)
scores

#%%
ev.roc(labels, scores)

#%%
scores = as_map
labels = gt_map

#%%
as_flatten = scores.ravel()
gt_flatten = labels.ravel()
#%%
ev.roc(gt_flatten, as_flatten)
#%%
my_range = (0, 0.03)

hist = plt.hist(as_flatten, bins=50, density=True,
                            range=my_range)

plt.plot([thr,thr], [0, max(hist[0])], c='r', marker='o')

#%%
a = plt.hist(as_flatten, bins=50, density=True, cumulative=True, range=my_range)
density = np.cumsum(n) / np.cumsum(n)[-1]

prob = 0.95

index = np.where(density >= prob)[0][0]

thr = bins[index]
