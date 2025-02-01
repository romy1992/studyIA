import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('classificazione_dataset_4.csv')
dataset.drop_duplicates(inplace=True)
le = LabelEncoder()
dataset['Genere'] = le.fit_transform(dataset['Genere'])
dataset['Esito'] = le.fit_transform(dataset['Esito'])

dataset.to_csv('film.csv', index=False)
