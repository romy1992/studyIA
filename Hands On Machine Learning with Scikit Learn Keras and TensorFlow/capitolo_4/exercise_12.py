# -*- coding: utf-8 -*-
"""
Created on Sat May 17 17:38:59 2025

@author: trotta

12.Implementare la discesa del gradiente batch con arresto anticipato per la regressione Softmax (senza utilizzare Scikit-Learn).

"""

# %%
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1,1)
# %%