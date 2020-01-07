# -*- coding: utf-8 -*-

#%%
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

from libraries.model.dataset import 
#%%

mask = np.array([0, 1.2 , 1.5, 2.3 ])
mask

gaussian_filter1d(mask, sigma=1)


