import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('classificazione_dataset_2.csv')
dataset.drop_duplicates()
le = LabelEncoder()
dataset['Spam'] = le.fit_transform(dataset['Spam'])
dataset.to_csv('email.csv', index=False)
