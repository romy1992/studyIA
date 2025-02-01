import joblib
import numpy as np
import pandas as pd

best_model = joblib.load('best_model_vendite.pkl')
scaler = best_model.named_steps['scaler']
model = best_model.named_steps['regressor']

dataset = pd.read_csv('vendite_test.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = scaler.transform(X)
predict = model.predict(X)
print(np.concatenate((predict.reshape(len(X), 1), y.reshape(len(y), 1)), axis=1))
