"""
Created on Tue May  6 17:20:19 2025
@author: trotta

1.	Provare un regressore Support Vector Machine (sklearn.svm.SVR) 
   con vari iperparametri, come kernel="linear" (con vari valori per l'iperparametro C) o kernel="rbf" (con vari valori per gli  iperparametri C e gamma). 
   Per il momento, non preoccupatevi del significato di questi iperparametri. Come si comporta il miglior predittore SVR?
   
2.	Prova a sostituire GridSearchCV con RandomizedSearchCV.
3.	Provare ad aggiungere un trasformatore nella pipeline di preparazione per selezionare solo gli attributi più importanti.
4.	Provare a creare una singola pipeline che esegua la preparazione completa dei dati e la stima finale.
5.	Esplora automaticamente alcune opzioni di preparazione utilizzando GridSearchCV.
"""
# %%
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
# %%


# %%
# Leggo il dataset
dataset = pd.read_csv("../housing/housing.csv")
# %%

# %%
# GridSearch
regressor = SVR()
params_grid_search = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]
grid_search=GridSearchCV(estimator=regressor,param_grid=params_grid_search,cv=5, scoring='neg_mean_squared_error',verbose=2)
# %%

# %%
# Split
X=dataset.drop(columns=['ocean_proximity']).values
y=dataset['median_house_value'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# %%

# %%
# StandrardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# %%

# %%
# Addestro
grid_search.fit(X_train, y_train)
# %%


# %%
# Score
negative_mse=grid_search.best_score_
rsme = np.sqrt(-negative_mse)
# %%


# %%
# Predict
#y_predict = grid_search.predict(X_test)
# %%

# %%
# Create pipeline
#pipeline = Pipeline([('scaler', sc) , ('model', grid_search.best_estimator_)])
# %%
