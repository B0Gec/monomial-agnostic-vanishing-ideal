# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt

from mavi.vanishing_ideal import VanishingIdeal
from mavi.util.preprocessing import Preprocessor


import sys
sys.path.append('..')
# import torch
import numpy as np
import matplotlib.pyplot as plt
from mavi.vanishing_ideal import VanishingIdeal
from mavi.util.preprocessing import Preprocessor
print(sys.executable)
theta = [np.pi*i/3 for i in range(6)]
X = np.vstack((np.cos(theta), np.sin(theta))).T
plt.plot(X[:,0], X[:,1], "o")
X = np.repeat(X, 1000, axis=0)
Z = np.repeat(X, 20, axis=1)
print(X.shape, Z.shape)
#%%time
# Preprocessing remove redundant variables 
pre = Preprocessor()
Z_ = pre.fit_transform(Z, th=.95, keep_dim=False)  # PCA-like preprocessing, 95% of power is kept.
print(Z_.shape)

# ### Comparison in VCA
# #%%time
# Computation for 2-dim dataset X
vi = VanishingIdeal()
vi.fit(X, 0.01, method="vca", backend='numpy')  

#%%time
'''
Computation for 40-dim dataset Z
'''
vi = VanishingIdeal()
vi.fit(Z, 0.01, method="vca", backend='numpy')  

#%%time
'''
Computation for preprocessed 2-dim dataset Z_
'''
vi = VanishingIdeal()
vi.fit(Z_, 0.01, method="vca", backend='numpy')  

### Comparison in Grad

#%%time
'''
Computation for 2-dim dataset X
'''
vi = VanishingIdeal()
vi.fit(X, 0.01, method="grad", backend='numpy')  

#%%time
'''
Computation for 40-dim dataset Z
'''
vi = VanishingIdeal()
vi.fit(Z, 0.01, method="grad", backend='numpy')  

#%%time
'''
Computation for preprocessed 2-dim dataset Z_
'''
vi = VanishingIdeal()
vi.fit(Z_, 0.01, method="grad", backend='numpy')  

### Plot

#%%time
'''
Don't forget to apply ```pre.transform```
'''
vi = VanishingIdeal()
vi.fit(Z_, 0.01, method="grad", backend='numpy')  
vi.plot(pre.transform(Z))
