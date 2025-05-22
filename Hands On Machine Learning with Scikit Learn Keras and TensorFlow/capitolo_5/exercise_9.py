# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:33:33 2025

@author: trotta

9.	Eseguire il training di un classificatore SVM sul set di dati MNIST. 
Poiché i classificatori SVM sono classificatori binari, 
sarà necessario utilizzare uno contro il resto per classificare tutte e 10 le cifre. 
È possibile ottimizzare gli iperparametri utilizzando piccoli set di convalida per velocizzare il processo. 
Quale precisione puoi raggiungere?

"""
# %%

from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]
# %%
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Con C=5 e loss
linear_svc=LinearSVC(C=5,loss='hinge',random_state=42)
linear_svc.fit(X_train, y_train)
y_pred=linear_svc.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)

# %% semplice Linear
linear_svc_2=LinearSVC(random_state=42)
linear_svc_2.fit(X_train, y_train)
y_pred_2=linear_svc_2.predict(X_test)
accuracy_2=accuracy_score(y_test, y_pred_2)
print(accuracy_2)
# %% con kernel 'RBF'
svc=SVC(kernel='rbf',random_state=42,gamma=3,C=5)
svc.fit(X_train, y_train)
y_pred_svc=svc.predict(X_test)
accuracy_svc=accuracy_score(y_test, y_pred_svc)
print(accuracy_svc)
# %% con Gamma 'scale'
svc_scale=SVC(gamma='scale')
svc_scale.fit(X_train, y_train)
y_pred_svc_scale=svc.predict(X_test)
accuracy_svc_scale=accuracy_score(y_test, y_pred_svc_scale)
print(accuracy_svc_scale)
# Si potrebbe anche aggiungere in GridSearch o un RandomSearch