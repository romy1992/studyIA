import pandas as pd

# Leggo
dataset = pd.read_csv('tema_dataset_2.csv')

# Controllo se ci sono duplicati
if dataset.duplicated().sum() > 0:
    dataset.drop_duplicates()

# Utilizzo il dummy per la colonna Materiale
dataset = pd.get_dummies(data=dataset, columns=['Materiale'], prefix=None, dtype=int)

# Salvo il nuovo dataset
dataset.to_csv('prezzi_energia.csv', index=False)
