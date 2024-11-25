import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

# Leggi il CSV
df = pd.read_csv('spese_mensili.csv')
df = df.drop('Mese', axis=1)

# X e y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Codifica delle feature
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = ct.fit_transform(X)
# X = np.array(X)
# X = X.reshape(len(X), 1)

# Dividi il set di dati in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# st = StandardScaler()
# X_train = st.fit_transform(X_train)
# X_test = st.fit_transform(X_test)

# Adatta il modello
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_predict = regressor.predict(X_test)

# Pylot
# X_train = X_train[:, -1].toarray()
# X_train = X_train.reshape(len(X_train), 1)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Previsioni stipendio mensile')
plt.xlabel('Ore')
plt.ylabel('Stipendio')
plt.show()
