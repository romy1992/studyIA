import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

"""
SVR cerca di trovare una funzione che approssima i dati riducendo al minimo gli errori, ma con una tolleranza (𝜖) 
che definisce una fascia ("epsilon-tube") entro cui gli errori sono accettabili.
Utilizza i support vectors (i dati più vicini o fuori dal margine) per determinare il modello di previsione.

Kernel RBF: SVR può essere lineare o non lineare. 
    Usando kernel come RBF (Radial Basis Function), SVR può adattarsi a relazioni complesse non lineari.

"""

# Leggo il csv
df = pd.read_csv('Position_Salaries.csv')

# Divido in X e y
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
# Riadatto in 2d con reshape inserendo la lunghezza delle righe(len(y)) e quante colonne(1)
y = y.reshape(len(y), 1)

# Controllo se ci sono valori null or OneHotEncoder da implementare (per questo esempio no)

# Split in train e test (per questo esempio no)


"""
Feature scaling di X e y = 
StandardScaler -> Serve per garantire che le variabili abbiano lo stesso peso durante l'addestramento, soprattutto 
    per modelli sensibili come SVR.
Applicati 2 StandardScaler sia per X e y perché ognuno di loro non deve dipendere dall'altro per formule interne
    tipo la medie e la derivazione di X deve essere diversa da y.
Se avessi usato la stessa allora y avrebbe avuto valori derivanti da X :
    Quando bisogna fare più scaling, bisogna creare quindi più oggetti per le colonne altrimenti il modello SVR farebbe 
    confusione.
"""
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)
"""
fit()-> Adatta il trasformatore ma non trasforma i dati
transform()-> Applica la formula e trasforma i dati
fit_transform()-> è l'unione dei 2
Per fare un esempio:
 X_scaled = X - μ(è la media calcolata durante fit()) / σ è la deviazione standard calcolata durante fit().
 Quindi μ e σ sono parametri che servono alla formula del transform

fit_transform():
Calcola i parametri (come la media e la deviazione standard, o qualsiasi altro parametro dipendente dal trasformatore) sui dati e applica la trasformazione nello stesso momento.
Si usa solo una volta, in genere sui dati di training.
Usalo solo sui dati di training o quando adatti e trasformi allo stesso tempo.

transform():
Applica la trasformazione sui dati senza ricalcolare i parametri, utilizzando quelli calcolati in precedenza con fit().
Si usa quando vuoi applicare la stessa trasformazione su dati diversi, come il test set o nuovi dati.
Usalo sui dati di test o su nuovi dati dopo che hai già calcolato i parametri sui dati di training.
"""

"""
Addestramento con rbf: Il kernel rfb serve per modelli non lineari e che non sono bene descritti da una semplice retta:
SVR è un tipo di modello che crea un tubo dove all'interno c'è una tolleranza di errori (che non seguono la retta e che
    NON vengono reputati proprio errori finché restano nel loro tubo)
"""
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

"""
Predict: 
inverse_transform(...) -> serve per ripristinare il valore iniziale dove prima era stato scalato in 0 o 1, e che ora 
    invertendolo, tornerà in euro/dollari . Senza questa trasformazione inversa, 
    il risultato della previsione sarà su una scala standardizzata 
    (media 0, deviazione standard 1), il che non ha senso in termini di stipendio reale.
"""
predict = scaler_y.inverse_transform(regressor.predict(scaler_X.transform([[6.5]])).reshape(-1, 1))

# Pyplot
plt.scatter(scaler_X.inverse_transform(X), scaler_y.inverse_transform(y), color='red')
plt.plot(scaler_X.inverse_transform(X), scaler_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title('Vero o Bluff')
plt.xlabel('Livello')
plt.ylabel('Salario')
plt.show()

# Pyplot Higher resolution
X_grid = np.arange(min(scaler_X.inverse_transform(X)), max(scaler_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(scaler_X.inverse_transform(X), scaler_y.inverse_transform(y), color='red')
plt.plot(X_grid, scaler_y.inverse_transform(regressor.predict(scaler_X.transform(X_grid)).reshape(-1, 1)), color='blue')
plt.title('Vero o Bluff')
plt.xlabel('Livello')
plt.ylabel('Salario')
plt.show()
