import joblib
import numpy as np
import pandas as pd

best_pipeline = joblib.load('best_model_medical.pkl')
sc = best_pipeline.named_steps['scaler']
model = best_pipeline.named_steps['classifier']

dataset = pd.read_csv('medical_test.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = sc.trasform(X)
y_predict = model.predict(X)
print(np.concatenate((y.reshape(len(y), 1), y_predict.reshape(len(y_predict), 1)), axis=1))

