import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('pima-indians-diabets.csv')

# Conta i valori a null
num_nan = df.isnull().sum()
print(f'Totali NaN :\n{num_nan}')

# Crea la media tra tutti le colonne mancanti.In questo caso non ci sono
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(df)
df_impute = impute.transform(df)

print(df_impute)
