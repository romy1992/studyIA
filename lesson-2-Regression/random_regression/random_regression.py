import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Leggo il csv
df = pd.read_csv('Position_Salaries.csv')

# Prendo X e y
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Controllare sempre se ci sono NaN o Encode da fare (Non per questo esempio)

# Scompattare in train e test i valori (Non per questo esempio)

# Riadattare i valori train prima dell'addestramento con il Feature Scaling (Non in questo esempio)

# Addestrare
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Fare la predizione
y_predict = regressor.predict([[6.5]])

# Pyplot alta soluzione
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""
RandomForestRegressor è un modello di regressione che utilizza un insieme di alberi decisionali per variabili continue.
Si basa sul concetto di alberi forestali e combina le loro previsioni per ridurre il rischio di overfitting.
Crea molti alberi decisionali (centinaia o migliaia) dove ognuno di loro effettua previsioni separati a differenza
dell'unico albero decisionale per ridurre il rischio di overfitting :
Parametri importanti del RandomForestRegressor:

n_estimators: Il numero di alberi nella foresta. Un numero più alto tende a migliorare la precisione, 
    ma aumenta anche il costo computazionale.
    
max_depth: La profondità massima di ogni albero. Limitarla può ridurre overfitting.

min_samples_split: Il numero minimo di campioni richiesti per dividere un nodo. Aumentarlo può ridurre overfitting.

min_samples_leaf: Il numero minimo di campioni che devono essere in una foglia. 
    Impostare un valore più alto impedisce la creazione di foglie con pochi campioni.

max_features: Il numero massimo di feature da considerare quando si divide un nodo. 
    Impostarlo su un valore più basso aiuta a creare alberi diversi, riducendo la correlazione tra gli alberi.
    
Differenza rispetto al DecisionTreeRegressor:
DecisionTreeRegressor costruisce un singolo albero decisionale, che può adattarsi troppo ai dati (overfitting).
RandomForestRegressor costruisce molti alberi, ognuno dei quali contribuisce alla previsione finale, riducendo 
    overfitting e migliorando la generalizzazione.(SARANNO MENO PRECISI PERO')
"""