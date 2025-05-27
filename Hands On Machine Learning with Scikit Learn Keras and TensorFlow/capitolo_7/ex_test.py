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

# %% oob (out-of-bag) -> Con algoritmi bagging, possiamo abilitare il parametro oob_score=True che sarà in grado di farci "evitare" il cv ,tts e altre valutazioni finali 

random_oob=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,random_state=42,oob_score=True)
random_oob.fit(X, y)
print(f"Accuratezza out-of-bag: {random_oob.oob_score_:.3f}")


random_no_oob=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,random_state=42)
random_no_oob.fit(X_train, y_train)
y_pred_no_oob=random_no_oob.predict(X_test)
print(f"Accuratezza senza out-of-bag: {accuracy_score(y_test, y_pred_no_oob)}")



# %% ADABOOST -> Sulla base di un Algoritmo, proverà ad addestrare e controllare le stime .Se alcune stime saranno errate, si riaddestrerà sulla base di quelle stime cercando di migliorare sui suoi errori
from sklearn.ensemble import AdaBoostClassifier

"""
algorithm='SAMME'	Usa solo le predizioni di classe (predict), senza probabilità.
È più "grezzo" e può convergere più lentamente.

algorithm='SAMME.R'	Usa le probabilità predette (predict_proba) dal classificatore base.
Questo rende il boosting più stabile e accurato. (Default)
"""
ada_clas=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=200,algorithm='SAMME.R',learning_rate=0.5,random_state=42)
ada_clas.fit(X_train, y_train)

y_pred_ada=ada_clas.predict(X_test)
print(accuracy_score(y_test, y_pred_ada))
# %% Gradient Boosting -> simile ad ADABOOST ma invece che correggere le stime del precedente, lui le riadatta, esempio:
import numpy as np
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X) # Notiamo che y2 sarà il riadattamento dell'y
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X) # Qui invece y3 si riadatta a y2
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

# * Infatti con GradientBoostingRegressor notiamo che il valore predetto è lo stesso di quello precedente 
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
test_y_pred=gbrt.predict(X_new)# *stesso valore di y_pred

gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
gbrt_slow.fit(X, y)
test_y_pred_2=gbrt_slow.predict(X_new)
"""
I 2 modelli sopra indicati hanno learning_rate diversi.Entrambi hanno dei problemi 
perchè il primo non si adatta bene (underfitting) il secondo si adatta troppo (overfitting).
Solitamente quando il learning_rate diminuisce, bisogna aumentare gli alberi (n_estimators).
Quindi possiamo cercare di bloccarlo prima o possiamo trovare gli alberi corretti usando il staged_predict() 
che restituisce le stime fatte per ogni addestramento : 
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)] # Crea una lista di errori utilizzando il MSE ciclato con lo staged_predict
bst_n_estimators = np.argmin(errors) + 1 # Dalla lista precedente prenderà l'errore MINORE

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42) # l'errore minore l'ha utilizzato come n_estimators che è quello che ha minor errore
gbrt_best.fit(X_train, y_train)
y_pred_best_gb=gbrt_best.predict(X_val)
mse=mean_squared_error(y_val, y_pred_best_gb)
"""
GradientBoostingRegressor è una versione più scolastica utilizzata per piccoli/medi dataset.
Utilizzare XGBoost che è l'evoluzione di GradientBoostingRegressor
"""