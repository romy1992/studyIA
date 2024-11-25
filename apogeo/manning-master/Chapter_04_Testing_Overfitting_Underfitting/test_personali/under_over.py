import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import make_classification

# Creiamo un dataset di esempio per la regressione
np.random.seed(42)
X = 2 * np.random.rand(100, 3)
y = 4 + 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LinearRegression
regressor_linear = LinearRegression()
regressor_linear.fit(X_train, y_train)
y_predict = regressor_linear.predict(X_test)
linear_mse = mean_squared_error(y_test, y_predict)
print(f"Mean Squared Error (Linear Regression): {linear_mse}")

# Lasso L1(Lasso Regression)
lasso_regressor = Lasso(alpha=0.1)
lasso_regressor.fit(X_train, y_train)
y_predict_lasso = lasso_regressor.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_predict_lasso)
print(f"Mean Squared Error (Lasso): {lasso_mse}")

# Ridge L2(Ridge Regression)
ridge_regressor = Ridge(alpha=1.0)
ridge_regressor.fit(X_train, y_train)
y_predict_ridge = ridge_regressor.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_predict_ridge)
print(f"Mean Squared Error (Ridge): {ridge_mse}")

# Regolarizzazione combinata L1 e L2 (ElasticNet)
elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train)
y_pred_elasticnet = elasticnet_model.predict(X_test)
elasticnet_mse = mean_squared_error(y_test, y_pred_elasticnet)
print(f"Mean Squared Error (ElasticNet): {elasticnet_mse}")

# Coefficienti dei modelli
print("Coefficiente modello Linear Regression:", regressor_linear.coef_)
print("Coefficiente modello Ridge:", ridge_regressor.coef_)
print("Coefficiente modello Lasso:", lasso_regressor.coef_)
print("Coefficiente modello ElasticNet:", elasticnet_model.coef_)

# Creiamo un dataset di esempio per la classificazione
X_class, y_class = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Regressione logistica con regolarizzazione L2 (default)
logistic_model_l2 = LogisticRegression(penalty='l2', solver='liblinear')
logistic_model_l2.fit(X_train_class, y_train_class)
y_pred_logistic_l2 = logistic_model_l2.predict(X_test_class)
accuracy_l2 = accuracy_score(y_test_class, y_pred_logistic_l2)
print(f"Accuracy (Logistic Regression con L2): {accuracy_l2}")

# Regressione logistica con regolarizzazione L1
logistic_model_l1 = LogisticRegression(penalty='l1', solver='liblinear')
logistic_model_l1.fit(X_train_class, y_train_class)
y_pred_logistic_l1 = logistic_model_l1.predict(X_test_class)
accuracy_l1 = accuracy_score(y_test_class, y_pred_logistic_l1)
print(f"Accuracy (Logistic Regression con L1): {accuracy_l1}")

"""
La regolarizzazione L1 e L2 vengono spesso utilizzate nei modelli di regressione per evitare l'overfitting,
aggiungendo dei termini di penalità alla funzione obiettivo. 
La differenza tra L1 e L2 è nel modo in cui viene applicata la penalizzazione ai coefficienti.

Regolarizzazione L1 (Lasso Regression): Aggiunge il valore assoluto dei coefficienti alla funzione obiettivo.
    Porterà i coefficienti a 0 e serve per modelli che hanno numerose caratteristiche e vogliamo trasformare
    alcune a 0.
Regolarizzazione L2 (Ridge Regression A CRESTA): Aggiunge il quadrato dei coefficienti alla funzione obiettivo
    Porterà i coefficienti QUASI vicino lo 0 e viene utilizzato per modelli di poche caratteristiche che vogliamo 
    renderle più piccole.Preferibilmente utilizzato per i Polynomial.
"""