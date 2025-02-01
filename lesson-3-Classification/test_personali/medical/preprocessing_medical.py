import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('classificazione_dataset_3.csv')
dataset.drop_duplicates(inplace=True)
le = LabelEncoder()
dataset['Stato di Salute'] = le.fit_transform(dataset['Stato di Salute'])
dataset.to_csv('medical.csv', index=False)
