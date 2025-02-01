from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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


def create_grid_search():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', None)
    ])
    grid_search = GridSearchCV(param_grid=params, estimator=pipeline, verbose=2, scoring='accuracy', cv=5)
    return grid_search
