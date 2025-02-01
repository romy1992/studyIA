import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


dataset = pd.read_csv('vendite.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = [
    # Linear Regression
    {
        'model': [LinearRegression()]  # Nessun iperparametro da ottimizzare per la regressione lineare
    },
    # Ridge Regression
    {
        'model': [Ridge()],
        'model__alpha': [0.01, 0.1, 1, 10, 100]  # Parametro di regolarizzazione
    },
    # Lasso Regression
    {
        'model': [Lasso()],
        'model__alpha': [0.01, 0.1, 1, 10, 100]  # Parametro di regolarizzazione
    },
    # Support Vector Regression (SVR)
    {
        'model': [SVR()],
        'model__C': [0.1, 1, 10, 100],
        'model__epsilon': [0.01, 0.1, 1],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']  # Solo per kernel rbf
    },
    # Random Forest Regressor
    {
        'model': [RandomForestRegressor()],
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    # Gradient Boosting Regressor
    {
        'model': [GradientBoostingRegressor()],
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 10]
    },
    # K-Nearest Neighbors Regressor (KNN)
    {
        'model': [KNeighborsRegressor()],
        'model__n_neighbors': [3, 5, 10],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    },
    # Decision Tree Regressor
    {
        'model': [DecisionTreeRegressor()],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
]

pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', None)
    ])

grid_search = GridSearchCV(param_grid=params, estimator=pipeline, verbose=2, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

print("Miglior modello", grid_search.best_estimator_)
print("Migliori parametri", grid_search.best_params_)
print("Miglior punteggio", grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict(X_test)


# Valutazione
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared (R²):", r2_score(y_test, y_pred))
print("Mean Absolute Percentage Error (MAPE):", mean_absolute_percentage_error(y_test, y_pred))


# Salva il modello
joblib.dump(grid_search.best_estimator_, 'best_model_vendite.pkl')
