import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Recupero il dataframe
df = pd.read_csv('50_Startups.csv')

# Identifico X e y
X = df.drop('Profit', axis=1).values
y = df['Profit'].values

# Controllo se ci sono valori NaN o Encoder da fare
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split dei dati per costruire il modello
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Addestro il modello
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predict test
y_predict = regression.predict(X_test)
# Per settare i valori numerici con 2 decimali
np.set_printoptions(precision=2)
# Concatenerà i valori predetti e quelli reali in verticale per confrontarli con più facilità
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

"""
La regressione lineare multipla è un'estensione della regressione lineare semplice e si utilizza quando si vuole 
prevedere una variabile dipendente continua (output) utilizzando più variabili indipendenti (input).
La regressione lineare multipla è utile quando il fenomeno che si vuole prevedere è influenzato da diversi fattori. 
Ad esempio:
Prezzo di una casa: Potresti volerlo prevedere considerando variabili come la superficie, 
    il numero di stanze, la posizione, l'anno di costruzione, ecc.
Rendimento scolastico: Potrebbe dipendere da fattori come le ore di studio, 
    il supporto familiare, il tipo di scuola, e così via.
Utilizzare più variabili indipendenti permette al modello di avere una maggiore capacità di previsione e di catturare 
meglio la complessità dei dati rispetto alla regressione lineare semplice, che si basa su una sola variabile.

Regressione lineare multipla: Si utilizzano più variabili indipendenti (𝑥1,𝑥2,...,𝑥𝑛) 
per prevedere la variabile dipendente (y). L'equazione diventa:
y = β0+β1x1+β2x2+...+βnXn+ϵ

dove:
y è la variabile dipendente.
x è la variabile indipendente.
β0 è l'intercetta.(è come una sorta di "valore di base" da cui il modello parte per poi aggiungere (o sottrarre) 
    gli effetti delle variabili indipendenti sul valore previsto di 𝑦)
β1,β2,...,βn sono i coefficienti di regressione associati alle rispettive variabili indipendenti.
ϵ è il termine di errore.
"""
