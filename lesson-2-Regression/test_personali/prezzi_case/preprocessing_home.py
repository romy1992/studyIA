import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Leggo il dataset
dataset = pd.read_csv('tema_dataset_1.csv')
if dataset.duplicated().sum() > 0:
    dataset.drop_duplicates()

# Label Encoder per la posizione della casa
le = LabelEncoder()
dataset['Posizione'] = le.fit_transform(dataset['Posizione'])

# Salvo il nuovo dataset
dataset.to_csv('prezzi_case.csv')


