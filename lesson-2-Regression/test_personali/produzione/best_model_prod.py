import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('produzione.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

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

grid_search = GridSearchCV(estimator=pipeline, param_grid=params, verbose=2, n_jobs=-1, cv=8,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print('Miglior modello', grid_search.best_estimator_)
print('Migliori parametri', grid_search.best_params_)
print('Miglior punteggio', grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict(X_test)

# Calcolo delle metriche
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R2 Score: {r2}")

# Salva il modello
joblib.dump(grid_search.best_estimator_, 'best_model_prod.pkl')
