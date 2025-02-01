import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('classificazione_dataset_1.csv')
dataset.drop_duplicates(inplace=True)
le = LabelEncoder()
dataset['Colore'] = le.fit_transform(dataset['Colore'])
dataset['Forma'] = le.fit_transform(dataset['Forma'])
dataset['Tipo di Frutta'] = le.fit_transform(dataset['Tipo di Frutta'])
dataset.to_csv('fruit.csv',index=False)
