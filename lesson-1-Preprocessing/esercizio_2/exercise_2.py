import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Read csv
df = pd.read_csv('titanic.csv')
# Indentify columns category
categorical_features = ['Sex', 'Pclass', 'Embarked']
# Adapted preprocessing
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
# Fit and Transform le colonne selezionate
X = ct.fit_transform(df)
# Tutto me lo trasforma in array
X = np.array(X)

# Output per y
le = LabelEncoder()
y = le.fit_transform(df['Survived'])

print(X)
print(y)
