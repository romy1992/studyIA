import joblib
import numpy as np
import pandas as pd

# Carico il modello prima salvato
model_pipeline = joblib.load('best_model_eng.pkl')

# Recupero lo scaler
sc = model_pipeline.named_steps['scaler']

# Recupero il miglio modello
model = model_pipeline.named_steps['model']

# Carico il nuovo dataset
dataset = pd.read_csv('nuovo_dataset_prezzi_energia.csv')

# X e y
X = dataset.drop(columns=['Consumo Energetico (kWh/m2)'], axis=1).values
y = dataset['Consumo Energetico (kWh/m2)'].values

# Faccio lo scaling dei dati nuovi recuperando il vecchio scaling
X = sc.transform(X)

# Faccio il predict
y_predict = model.predict(X)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y.reshape(len(y), 1)), 1))
