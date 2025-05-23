# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:52:12 2025

@author: trotta

ESEMBLE E RANDOM FOREST
"""
# %% VotingClassifier -> è un accumulatore di algoritmi in grado di dare una valutazione finale più robusta
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# %% HARD
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

random_forest=RandomForestClassifier(random_state=42,n_estimators=100)
logistic=LogisticRegression(solver='lbfgs',random_state=42)
svc=SVC(gamma='scale',random_state=42)

# Iperparametro voting='hard' aggrega tutte le stime dei classificatori e restituisce la classe con il maggior numero di voti
voting = VotingClassifier(estimators=[('rf',random_forest),('lg',logistic),('svc',svc)],voting='hard')
voting.fit(X_train, y_train)
y_pred=voting.predict(X_test)
accuracy_voting=accuracy_score(y_test, y_pred)

for model in (random_forest, logistic, svc, voting):
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    print(model.__class__.__name__,accuracy_score(y_test, y_predict))
# %% SOFT
random_forest=RandomForestClassifier(random_state=42,n_estimators=100)
logistic=LogisticRegression(solver='lbfgs',random_state=42)
svc=SVC(gamma='scale',random_state=42,probability=True)# abititare l'iperparametro probality a True per il predict_proba che è idoneo per il voto soft

# Iperparametro voting='soft' stima invece la probabilità , il che lo rende con stime più alte e più attendibili
voting = VotingClassifier(estimators=[('rf',random_forest),('lg',logistic),('svc',svc)],voting='soft')
voting.fit(X_train, y_train)
y_pred=voting.predict(X_test)
accuracy_voting=accuracy_score(y_test, y_pred)

for model in (random_forest, logistic, svc, voting):
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    print(model.__class__.__name__,accuracy_score(y_test, y_predict))
# %% BAGGING and PASTING (Insaccamento e incollaggio) 
"""
Bagging (bootstrap aggregating) → consiste nell’addestrare ciascun modello su un sottoinsieme casuale con sostituzione del dataset di training. Quindi ogni campione può essere scelto più volte.

Pasting (incollaggio) → come il bagging, ma senza sostituzione. Ogni istanza può essere scelta al massimo una volta per ogni modello.
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_jobs=-1,
    n_estimators=500,
    max_samples=100,
    bootstrap=True,# Con true (BAGGING) userà il voto 'soft' se il classificatore dispone di predict_proba altrimenti sarà un 'hard'.Con false invece questo diventarà PASTING
    random_state=42)
bag_clf.fit(X_train, y_train)
y_pred_bag=bag_clf.predict(X_test)
accuracy_bag=accuracy_score(y_test, y_pred_bag)
# %% RANDOM FOREST -> FORESTE CAUSALI CHE INCLUDONO IL BAGGING E PASTING QUINDI E' SOSTITUIBILE(entrambi basati sul bagging)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
random=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,random_state=42)
random.fit(X_train, y_train)
y_pred_rf=random.predict(X_test)
print(random.feature_importances_)#restituisce le caratteristiche con maggiore importanza

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, random_state=42)

bag_clf.fit(X_train, y_train)
y_pred_bag_clf = bag_clf.predict(X_test)

np.sum(y_pred_bag_clf == y_pred_rf) / len(y_pred_bag_clf) # Saranno entrambi con lo stesso risultato

