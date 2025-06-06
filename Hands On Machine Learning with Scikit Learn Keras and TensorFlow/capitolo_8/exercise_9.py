# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:30:17 2025

@author: trotta

9.	
Caricare il set di dati MNIST (introdotto nel Capitolo 3) e dividerlo in un set di addestramento e un set di test (prendere le prime 60.000 istanze per il training e le restanti 10.000 per il test).

Eseguire il training di un classificatore Random Forest sul set di dati e calcolare il tempo necessario, quindi valutare il modello risultante sul set di test.

Successivamente, utilizza la PCA per ridurre la dimensionalità del set di dati,con un rapporto di varianza spiegato del 95%. 

Eseguire il training di un nuovo classificatore Random Forest sul set di dati ridotto e vedere quanto tempo ci vuole. 

L'allenamento è stato molto più veloce? Valutare quindi il classificatore nel set di test. Come si confronta con il classificatore precedente?
# %%
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
mnist = fetch_openml('mnist_784', version=1,as_frame=False)

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random_forest=RandomForestClassifier(n_estimators=200, random_state=42)
random_forest.fit(X_train, y_train)
# %%
y_pred=random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# %%
from sklearn.decomposition import PCA

pca=PCA(n_components=0.95)
X_trasform=pca.fit_transform(X_train)
# %%
random_forest_pca=RandomForestClassifier(n_estimators=200, random_state=42)
random_forest_pca.fit(X_trasform, y_train)
# %%
X_test=pca.transform(X_test)
y_pred_pca=random_forest_pca.predict(X_test)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

"""
Stesso tempo, quindi non sempre PCA riduce i tempi
Senza PCA 0.9688
Con PCA 0.9492
Quindi qualcosa con PCA l'ha persa
MORALE : Non sempre la ridimensionalità riduce i tempi e migliora le prestazioni
"""