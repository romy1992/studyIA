# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:09:45 2025

@author: trotta

8.	Caricare i dati MNIST (introdotti nel Capitolo 3) e dividerli in un set di addestramento, 
un set di convalida e un set di test (ad esempio, utilizzare 50.000 istanze per l'addestramento, 10.000 per la convalida e 10.000 per il test). 
Eseguire quindi il training di vari classificatori, ad esempio un classificatore Random Forest, un classificatore Extra-Trees e un classificatore SVM. 
Successivamente, prova a combinarli in un insieme che superi ogni singolo classificatore nel set di convalida, utilizzando il voto soft o hard. 
Una volta trovato uno, provalo sul set di test. Quanto è più performante rispetto ai singoli classificatori?

"""
# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
mnist = fetch_openml('mnist_784', version=1,as_frame=False)

X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)
# %%
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import LinearSVC

random_forest=RandomForestClassifier(random_state=42,n_estimators=100)
extra_forest=ExtraTreesClassifier(random_state=42,n_estimators=100)
linear_svc=LinearSVC(random_state=42,max_iter=100)

models=[random_forest,extra_forest,linear_svc]
for model in models:
    print('Training',model)
    model.fit(X_train, y_train)
    
# %%
print([model.score(X_val, y_val) for model in models])
# %%
from sklearn.ensemble import VotingClassifier

voting=VotingClassifier(estimators=[
    ('random_forest',random_forest),
    ('extra_forest',extra_forest),
    ('linear_svc',linear_svc)],verbose=2, n_jobs=-1)

voting.fit(X_train,y_train)
print(voting.score(X_val, y_val))
# %% Non c'è bisogno di riaddestrare per settare il voto hard o soft
voting.voting='hard' 
print('Hard',voting.score(X_val, y_val))

# LinearSVC non contiene il predict_proba perchè non gliel'ho abilitato quindi lo elimino dagli estimatori addestrati
voting.set_params(linear_svc=None) 
del voting.estimators_[2]

voting.voting='soft'
print('Soft',voting.score(X_val, y_val))
# %% 
import numpy as np
"""
9.	Eseguire i singoli classificatori dell'esercizio precedente per eseguire stime sul set di convalida e creare un nuovo set di training con le stime risultanti: 
    ogni istanza di training è un vettore contenente il set di stime di tutti i classificatori per un'immagine e la destinazione è la classe dell'immagine. 
Eseguire il training di un classificatore su questo nuovo set di training. Congratulazioni, hai appena addestrato un frullatore e, insieme ai classificatori, forma un insieme di impilamento! Ora valuta l'insieme sul set di test. 
Per ogni immagine nel set di test, eseguire stime con tutti i classificatori, quindi inviare le stime al frullatore per ottenere le stime dell'insieme. 
Come si confronta con il classificatore di voto che hai addestrato in precedenza?
"""
X_val_predict=np.empty((len(X_val), len(models)), dtype=np.float32)

for index,model in enumerate(models):
    X_val_predict[:,index]=model.predict(X_val)
# Riaddresto un nuovo modello con SOLO le valutazioni(predizioni) dei precedenti (Meta model) con i dati di validazione
random_f=RandomForestClassifier(n_estimators=200,random_state=42,oob_score=True)
random_f.fit(X_val_predict, y_val)
y_pred_random=random_f.oob_score_

# %%

X_test_predict=np.empty((len(X_test), len(models)), dtype=np.float32)

for index,model in enumerate(models):
    X_test_predict[:,index]=model.predict(X_test)
    
# Con i dati di test
y_pred_test=random_f.predict(X_test_predict)

from sklearn.metrics import accuracy_score
y_pred_random_accuracy=accuracy_score(y_test, y_pred_test)
print(y_pred_random_accuracy)
# %% EXTRA -> L'ESERCIZIO PRECEDENTE MOSTRA ENSEMBLE CON STACKING FATTO MANUALMENTE : SKLEARN HA PERò RILASCIATO UNA NUOVA VERSIONE CON ALGORITMI StackingClassifier e StackingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score

random_forest=RandomForestClassifier(random_state=42,n_estimators=100)
extra_forest=ExtraTreesClassifier(random_state=42,n_estimators=100)
estimators=[
    ('random_forest',random_forest),
    ('extra_forest',extra_forest)]

# NUOVA VERSIONE PER STACKING
stacking_cls=StackingClassifier(
    estimators=estimators, # Gli algoritmi che vuoi che addestrino in maniera sequenziale
    final_estimator=LogisticRegression(), # Il meta model che addestrerà sulla base dei valori predetti precedentemente dagli estimator
    cv=5 ,verbose=2, n_jobs=-1)# Addestrati tramite Cross Validation

stacking_cls.fit(X_train, y_train)
y_pred_stack=stacking_cls.predict(X_test)
accuracy_stacking=accuracy_score(y_test, y_pred_stack)
# %%
"""
import numpy as np
X_test_predict_stacking=np.empty((len(X_train), len(estimators)), dtype=np.float32)
for index,(_,model) in enumerate(estimators):
    model.fit(X_train,y_train)
    X_test_predict_stacking[:,index]=model.predict(X_train)
    
random_f_stack=RandomForestClassifier(n_estimators=200,random_state=42)
random_f_stack.fit(X_test_predict_stacking, y_train)
y_pred_random_stacking=random_f_stack.predict(X_test)
y_pred_random_accuracy=accuracy_score(y_test,y_pred_random_stacking)
"""