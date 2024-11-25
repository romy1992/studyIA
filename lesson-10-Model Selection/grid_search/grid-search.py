import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataframe = pd.read_csv('../Social_Network_Ads.csv')

X = dataframe.iloc[:, : -1].values
y = dataframe.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Cross value
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())

# Grid Search
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_
print(best_accuracy)
print(best_params)
"""
GridSearch è una griglia di valori la quale l'algoritmo e i parametri vengono utilizzati utilizzando una vasta gamma
di valori e addestrando i solito modelli di train.
Quello che restituirà sono accuracy più alte,e quindi meglio lavorate, e i parametri utilizzati per quel determinato
risultato.
"""
