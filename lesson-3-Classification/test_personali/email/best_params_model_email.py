import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report, \
    recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

dataset = pd.read_csv('email.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

params = [
    # Support Vector Machine
    {
        'classifier': [SVC(probability=True)],
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'classifier__degree': [2, 3, 4],  # Usato solo per kernel 'poly'
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    },
    # Random Forest
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200, 500],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__bootstrap': [True, False]
    },
    # Logistic Regression
    {
        'classifier': [LogisticRegression()],
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2', 'l1', 'elasticnet', 'none'],
        'classifier__solver': ['lbfgs', 'saga', 'liblinear'],  # Solver compatibili
        'classifier__max_iter': [100, 200, 500]
    },
    # K-Nearest Neighbors
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 7, 10],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['minkowski', 'euclidean', 'manhattan'],
        'classifier__p': [1, 2]  # Distanza L1 o L2
    },
    # Gradient Boosting (XGBoost)
    {
        'classifier': [XGBClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'classifier__max_depth': [3, 5, 10],
        'classifier__subsample': [0.5, 0.7, 1.0],
        'classifier__colsample_bytree': [0.5, 0.7, 1.0]
    },
    # Gradient Boosting (LightGBM)
    # {
    #     'classifier': [LGBMClassifier()],
    #     'classifier__n_estimators': [50, 100, 200],
    #     'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
    #     'classifier__max_depth': [-1, 5, 10],
    #     'classifier__num_leaves': [31, 50, 100],
    #     'classifier__subsample': [0.5, 0.7, 1.0],
    #     'classifier__colsample_bytree': [0.5, 0.7, 1.0]
    # },
    # Naive Bayes (Gaussian)
    {
        'classifier': [GaussianNB()],
        'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    },
    # Naive Bayes (Multinomial)
    {
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.1, 0.5, 1.0, 2.0],
        'classifier__fit_prior': [True, False]
    },
    # Decision Tree
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__criterion': ['gini', 'entropy', 'log_loss'],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': [None, 'sqrt', 'log2']
    }
]

grid_search = GridSearchCV(estimator=pipeline, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print('Best model', grid_search.best_estimator_)
print('Best score', grid_search.best_score_)
print('Best params', grid_search.best_estimator_)

y_pred = grid_search.predict(X_test)

# Calcolo e stampa delle metriche
print("Accuracy:", accuracy_score(y_test, y_pred))

# Precision, Recall e F1 (multiclasse gestito con 'macro')
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1-Score (macro):", f1_score(y_test, y_pred, average='macro'))

# Matrice di confusione e report completo
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Salva il modello
joblib.dump(grid_search.best_estimator_, 'best_model_email.pkl')
