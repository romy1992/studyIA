import joblib
import numpy as np
import pandas as pd

best_model = joblib.load('best_model_prod.pkl')
sc = best_model.named_steps['scaler']
model = best_model.named_steps['model']

dataset = pd.read_csv('produzione_test.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = sc.transform(X)

predict = model.predict(X)

print(np.concatenate((predict.reshape(len(predict), 1), (y.reshape(len(y), 1))), axis=1))
