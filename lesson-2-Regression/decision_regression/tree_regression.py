import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Leggiamo i dati
df = pd.read_csv('Position_Salaries.csv')

# Dividiamo per X e y
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Controllare se i dati hanno valori nulli o Encode da fare (non per questo esempio)

# Dividere per train_test_split (non per questo esempio)

# Creare il suo Albero decisionale per addestrare
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Dopo l'addestramento e prima della previsione, bisogna capire se i dati hanno bisogno di essere scalati(NO QUESTO ES.)

# Previsione: Solitamente nella previsione si dovrebbe inserire un X adatto per restituire i giusti risultati
y_predict = regressor.predict([[6.5]])
print(y_predict)

# Pyplot
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""
DecisionTreeRegressor : è un algoritmo basato sugli alberi decisionali utilizzato per variabili continue *.
                Ha una struttura ad albero basata su decisioni binarie ripetute sui dati 
                per segmentare lo spazio delle feature in regioni, all'interno delle quali le previsioni vengono fatte.


*Le variabili continue sono quei numeri infiniti , ovvero quei numeri interi o decimali , esempio:
Le temperature : 30.5 , 30 ecc
L'altezza : 1.72
Il peso : 58.3
Il salario : 1547
Quindi tutti quei numeri che hanno infinite possibilità di calcolo  

E poi ci sono quelle DISCRETE : 
sono numeri specifici e limitati come ad esempio il numero di persone in una stanza o il risultato di una partita.

"""
