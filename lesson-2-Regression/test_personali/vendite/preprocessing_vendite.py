import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('tema_dataset_3.csv')
if dataset.duplicated().sum() > 0:
    dataset.drop_duplicates()

le = LabelEncoder()
dataset['Posizione'] = le.fit_transform(dataset['Posizione'])

dataset.to_csv('vendite.csv', index=False)
