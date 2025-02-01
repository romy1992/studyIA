import pandas as pd

dataset = pd.read_csv('tema_dataset_4.csv')
dataset.drop_duplicates()
dataset.to_csv('produzione.csv', index=False)
