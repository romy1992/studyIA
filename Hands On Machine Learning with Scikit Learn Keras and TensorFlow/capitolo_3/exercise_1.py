# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:43:04 2025

@author: trotta

1.	Provare a creare un classificatore per il set di dati MNIST che raggiunga un'accuratezza superiore al 97% nel set di test. 
Suggerimento: il KNeighborsClassifier funziona abbastanza bene per questo compito; devi solo trovare buoni valori di iperparametro 
(prova una ricerca su griglia sui pesi e n_neighbors iperparametri).

"""

# %%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,as_frame=False)
X, y = mnist["data"], mnist["target"]
# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split

params=[{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=42)
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params,cv=5,verbose=3)
grid_search.fit(X_train,y_train)
# %%
print(grid_search.best_params_)
# %%
y_predict=grid_search.predict(X_test)
# %%
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_predict)


