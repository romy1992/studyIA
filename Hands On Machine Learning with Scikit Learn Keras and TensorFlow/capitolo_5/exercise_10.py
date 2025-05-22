# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:33:52 2025

@author: trotta

10.	Addestrare un regressore SVM sul set di dati sulle abitazioni della California.

"""
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)
# %%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
# %%
from sklearn.svm import LinearSVR
linear=LinearSVR(random_state=42)
linear.fit(X_scaled, y_train)
y_pred=linear.predict(X_test)
# %%
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred)
# %%
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform # ottimo per valori casuali ordinati

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
random=RandomizedSearchCV(SVR(), param_distributions,random_state=42,verbose=2,cv=3,n_iter=10)
random.fit(X_scaled, y_train)
# %%
import numpy as np
random.best_estimator_
mse_svr=mean_squared_error(y_test, y_pred)
np.sqrt(mse_svr)