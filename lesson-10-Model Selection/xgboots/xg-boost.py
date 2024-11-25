import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Data.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Addestro
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predict
y_predict = classifier.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

# Cross value
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())

# Grid Search
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(param_grid=parameters, cv=10, estimator=classifier, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
"""
XGBoots è un altro algoritmo di ML ma che fa parte del pacchetto di xgboots.
E' utile per maggiori accuratezze sia per modelli di classificazione (XGBClassifier) che regressione (XGBRegressor).

"E' una libreria di machine learning basata sull'algoritmo di gradient boosting progettata per essere altamente 
efficiente, flessibile e veloce.È particolarmente popolare per risolvere problemi di classificazione e regressione 
e viene spesso utilizzata in competizioni di machine learning per la sua alta accuratezza e prestazioni."
"""
