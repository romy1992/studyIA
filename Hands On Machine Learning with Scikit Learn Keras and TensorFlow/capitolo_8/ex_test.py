# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:30:16 2025

@author: trotta

Riduzione delle dimensionalità 
"""
# %% PCA
import numpy as np

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X2D = pca.fit_transform(X)

pca.explained_variance_ratio_
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1