import joblib
import numpy as np
import pandas as pd

best_pipeline = joblib.load('best_model_auto.pkl')
sc = best_pipeline.named_steps['scaler']
model = best_pipeline.named_steps['model']

dataset = pd.read_csv('vendite_auto_test.csv')
X = dataset.drop(columns=['Prezzo (Euro)'], axis=1).values
y = dataset['Prezzo (Euro)'].values

X = sc.transform(X)
prediction = model.predict(X)

print(np.concatenate((prediction.reshape(len(prediction), 1), (y.reshape(len(y), 1))), axis=1))
