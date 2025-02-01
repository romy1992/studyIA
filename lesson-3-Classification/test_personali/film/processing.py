import joblib
import numpy as np
import pandas as pd

best_pipeline = joblib.load('best_model_film.pkl')
sc = best_pipeline.named_steps['scaler']
model = best_pipeline.named_steps['classifier']

dataset = pd.read_csv('film_test.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = sc.transform(X)
y_pred = model.predict(X)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), (y.reshape(len(y), 1))), axis=1))
