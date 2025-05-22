# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:11:36 2025

@author: trotta

7 - Eseguire il training e ottimizzare un albero decisionale per il set di dati delle lune seguendo questa procedura:
a. Usa make_moons(n_samples=10000, noise=0.4) per generare un dataset di lune.
b. Utilizzare train_test_split() per suddividere il set di dati in un set di training e un set di test.
c. Utilizzare la ricerca a griglia con convalida incrociata (con l'aiuto della  classe GridSearchCV) per trovare valori di iperparametro validi per un DecisionTreeClassifier. 
                                                             Suggerimento: prova vari valori per max_leaf_nodes.
d. Esegui il training sull'intero set di training usando questi iperparametri e misura le prestazioni del tuo modello sul set di test. 
    Dovresti ottenere una precisione compresa tra l'85% e l'87%.

"""
# %% a
from sklearn.datasets import make_moons

X,y = make_moons(n_samples=10000, noise=0.4,random_state=42)
# %% b
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# %% c
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
params={'max_leaf_nodes':list(range(2,100)),'min_samples_split':[2,3,4]}
grid_search=GridSearchCV(DecisionTreeClassifier(random_state=42), params,cv=3,verbose=1)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_
grid_search.best_score_
grid_search.best_params_
# %% d
from sklearn.metrics import accuracy_score
y_pred=grid_search.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)
# %%
"""
8 - Fai crescere una foresta seguendo questi passaggi:
a.      Continuando l'esercizio precedente, generare 1.000 sottoinsiemi del set di addestramento, ciascuno contenente 100 istanze selezionate in modo casuale. 
        Suggerimento: puoi usare la  classe ShuffleSplit di ScikitLearn  per questo.
b.      Addestrare un albero decisionale su ogni sottoinsieme, utilizzando i migliori valori di iperparametro trovati nell'esercizio precedente. 
        Valutare questi 1.000 alberi decisionali nel set di test. Poiché sono stati addestrati su set più piccoli, questi alberi decisionali probabilmente avranno prestazioni peggiori rispetto al primo albero decisionale, raggiungendo solo circa l'80% di precisione.
c.      Ora arriva la magia. Per ogni istanza del set di test, generare le previsioni dei 1.000 alberi decisionali e mantenere solo la previsione più frequente (è possibile utilizzare la  funzione mode() di SciPy  per questo). 
        Questo approccio fornisce stime del voto di maggioranza sul set di test.
d.      Valuta queste previsioni sul set di test: dovresti ottenere un'accuratezza leggermente superiore rispetto al tuo primo modello (circa 0,5-1,5% in più). 
        Congratulazioni, hai addestrato un classificatore Random Forest!

"""
# a
from sklearn.model_selection import ShuffleSplit # ->La classe ShuffleSplit di scikit-learn serve per generare suddivisioni casuali (shuffle) del dataset in training e test set, utile per la validazione incrociata (cross-validation).

n_trees = 1000
n_instances = 100
sf=ShuffleSplit(n_splits=n_trees,test_size=len(X_train)-n_instances,random_state=42)
mini_set=[]
for mini_train_index,mini_test_index in sf.split(X_train):
    X_train_mini=X_train[mini_train_index]
    y_train_mini=y_train[mini_train_index]
    mini_set.append((X_train_mini,y_train_mini))

# %% b
from sklearn.base import clone
import numpy as np
forest=[clone(grid_search.best_estimator_) for _ in range(n_trees)]#clona 1000 alberi decisionali con il best estimator trovato in precedenza

accuracy_scores=[]
for tree,(X_mini_train,y_mini_train) in zip(forest,mini_set): # è un ciclo for tipo con mappe dove posso ciclare più liste contenenti in zip()
    tree.fit(X_mini_train,y_mini_train)
    y_pred=tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
np.mean(accuracy_scores)
# %% c
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
for tree_index,tree in enumerate(forest):
    Y_pred[tree_index]=tree.predict(X_test)

from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)# il mode() recupera la previsione della classe maggiore per ogni riga(y_pred_majority_votes)-con n_votes invece restituisce il numero di volte che quella classe ha vinto
# %% d
resh= y_pred_majority_votes.reshape([-1])
accuracy_score(y_test,resh)

