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
"""
Modalità per cambiare un risultato :
Basta sistemarte la soglia (threshold)
Solitamente la soglia è impostata a 0 e non c'è modo di aumentarla con i parametri
decision_function infatti ti restituisce la soglia del parametro predetto
"""
y_scores=sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)
# %%
"""
cross_val_predict inoltre ha anche un parametro "method" che con valore "decision_function" ,
restituisce tutte le soglie del modello predetto
"""
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method='decision_function')
print(y_scores)
# %%
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
"""
Grazie a precision_recall_curve, possiamo calcolare la precisione e il recall controllando tutte le loro soglie
"""
print(precisions, recalls, thresholds)
# %%
"""
Infine, utilizzare Matplotlib per tracciare la precisione e richiamare in funzione il valore di soglia
"""
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown



recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


plt.figure(figsize=(8, 4))                                                                  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Not shown
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown
save_fig("precision_recall_vs_threshold_plot")                                              # Not shown
plt.show()
# %%
(y_train_pred == (y_scores > 0)).all()
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
save_fig("precision_vs_recall_plot")
plt.show()

# %%
# (np.argmax() ti darà il primo indice del valore massimo, che in questo caso significa il primo  valore Vero):
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
# creo quindi un nuovo arrai con la soglia impostata sopra
y_train_pred_90 = (y_scores >= threshold_90_precision) 
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))
"""
I risultati , dopo aver cambiato la soglia, cambiano fino ad arrivare ad un 90% di precisione ma calano di recall
"""
# %%
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                                    # Not shown
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           # Not shown
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   # Not shown
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Not shown
plt.plot([fpr_90], [recall_90_precision], "ro")               # Not shown
save_fig("roc_curve_plot")                                    # Not shown
plt.show()
# %%
from sklearn.metrics import roc_auc_score 
print(roc_auc_score(y_train_5, y_scores))
# %% Multiclasse
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000]) # y_train, not y_train_5 -> y_train a differenza di y_train_5, contiene classi da 0 a 9 e quindi il classificatore "capirà" meglio in questo caso
svc_predict=svm_clf.predict([some_digit])
print(svc_predict)
some_digit_scores = svm_clf.decision_function([some_digit])#-> di fatti questo restituisce la "soglia" di ogni classe e il numero 5 avrà un numero elevato
print(some_digit_scores)
print(np.argmax(some_digit_scores))#np.argmax restituisce il numero più alto nell'array
print(svm_clf.classes_)# restituisce le classi analizzate

