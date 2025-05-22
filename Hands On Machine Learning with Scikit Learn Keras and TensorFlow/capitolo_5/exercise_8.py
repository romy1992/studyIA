# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:33:01 2025

@author: trotta

8.Addestrare un LinearSVC su un set di dati linearmente separabile. 
Eseguire quindi il training di un SVC e di un SGDClassifier sullo stesso set di dati. 
Vedi se riesci a farli produrre all'incirca lo stesso modello.

"""
# %%
from sklearn import datasets

iris=datasets.load_iris()
X=iris['data'][:,(2,3)]
y=iris['target']
setosa_or_versicolor = (y == 0) | (y == 1)
X=X[setosa_or_versicolor]
y=y[setosa_or_versicolor]
# %%
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

linear_svc=LinearSVC(C=5,loss="hinge",random_state=42)
svc=SVC(C=5,kernel='linear')
sgd_classifier=SGDClassifier(loss="hinge", learning_rate="constant", 
                             eta0=0.001, alpha=1/(5 * len(X)),
                        max_iter=1000, tol=1e-3, random_state=42)

scaler=StandardScaler()
X_scaler=scaler.fit_transform(X)

linear_svc.fit(X_scaler,y)
svc.fit(X_scaler,y)
sgd_classifier.fit(X_scaler,y)
# %%
print("Linear SCV ",linear_svc.intercept_,linear_svc.coef_)
print("SCV ",svc.intercept_,svc.coef_)
print("SGDClassifier ",sgd_classifier.intercept_,sgd_classifier.coef_)