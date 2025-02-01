import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('classificazione_dataset_5.csv')
dataset.drop_duplicates(inplace=True)
le = LabelEncoder()
dataset['Tipo di Veicolo'] = le.fit_transform(dataset['Tipo di Veicolo'])
dataset.to_csv('macchine.csv', index=False)
