# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:30:46 2025

@author: trotta
"""
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./housing/housing.csv')
dataset.head()

# dataset.corr()

dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,s=dataset["population"]/100, label="population", figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True) 
plt.legend()
