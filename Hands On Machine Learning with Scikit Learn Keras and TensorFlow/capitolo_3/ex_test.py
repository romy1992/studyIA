# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:42:33 2025

@author: trotta
"""

# %%
from sklearn.datasets import fetch_openml
from setup import save_fig
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mnist=fetch_openml('mnist_784', version=1,as_frame=False)
mnist.keys()

# %%
X,y=mnist['data'],mnist['target']
X.shape
y.shape

# %%
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

save_fig("some_digit_plot")
plt.show()

# %%
y[0]
y = y.astype(np.uint8)


# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# %%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

#%%
sgd_clf.predict([some_digit])


# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 
"""
from sklearn.base import clone :
    
Il clone non fa altro che clonare il modello di addestramento
(ha la stessa valenza di creare manualmente il modello come facevo io)
Il clone evita la duplicazione del codice
"""

"""
StratifiedKFold -> ha la "stessa valenza" del cross_val_score() : 
la differenza sta nel fatto che StratifiedKFold è più flessibile per questioni di iperparametri o operazioni manuali
"""
skfold=StratifiedKFold(n_splits=3,shuffle=True ,random_state=42)

# %%
for train_index , test_index in skfold.split(X_train, y_train_5):
    clone_clf=clone(sgd_clf)
    X_train_fold,y_train_fold,X_test_fold,y_test_fold = X_train[train_index], y_train_5[train_index],X_train[test_index], y_train_5[test_index]
    clone_clf.fit(X_train_fold, y_train_fold)
    y_predict=clone_clf.predict(X_test_fold)
    n_correct=sum(y_predict==y_test_fold)
    print(n_correct/len(y_predict))    
# %%
from sklearn.model_selection import cross_val_score
score=cross_val_score(sgd_clf, X_train, y_train_5, cv=3,scoring='accuracy')
print(score)
# %%
"""
Proprio come la  funzione cross_val_score(), cross_val_predict() esegue la convalida incrociata K-fold, 
ma invece di restituire i punteggi di valutazione, restituisce le stime effettuate su ogni riduzione del test. 
Ciò significa che si ottiene una stima pulita per ogni istanza nel set di addestramento 
("pulita" significa che la stima viene effettuata da un modello che non ha mai visto i dati durante l'addestramento).
"""
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(y_train_pred)
# %%
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train_5, y_train_pred)
print(cm)
# %%
from sklearn.metrics import precision_score, recall_score
ps=precision_score(y_train_5, y_train_pred)
print(ps)
rs=recall_score(y_train_5, y_train_pred)
print(rs)
# F1 unisce precisione e recall
from sklearn.metrics import f1_score
f1=f1_score(y_train_5, y_train_pred)
print(f1)
# %%
y_scores=sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)
# %%
y_train_decision = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method='decision_function')
print(y_train_decision)
# %%
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_train_decision)
print(precisions, recalls, thresholds)
# %%
import matplotlib.pyplot as plt 
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")     
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")  # highlight the threshold and add the legend, axis label, and grid
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds) 
    plt.show()
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
