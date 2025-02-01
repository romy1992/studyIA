import pandas as pd

dataset = pd.read_csv('tema_dataset_5.csv')
dataset.drop_duplicates()
dataset = pd.get_dummies(data=dataset, columns=['Marca'], dtype=int)
dataset.to_csv('vendite_auto.csv', index=False)
