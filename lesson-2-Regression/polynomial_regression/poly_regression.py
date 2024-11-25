import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def pyplot_regressor(**kwargs):
    X = kwargs.get('x')
    X_poly = kwargs.get('X_poly')
    y = kwargs.get('y')
    regressor = kwargs.get('regressor')
    title = kwargs.get('title')
    title_x = kwargs.get('title_x')
    title_y = kwargs.get('title_y')

    plt.scatter(X, y, color='red')
    plt.plot(X, regressor.predict(X_poly if X_poly is not None else X), color='blue')
    plt.title(title)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.show()


# Leggo i dati
df = pd.read_csv('Position_Salaries.csv')

# Divido i dati
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Addestriamo con LinearRegression senza fare lo split
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Addestriamo con Poly con scelta dell'esponente:
# è usata per trasformare le variabili indipendenti (𝑋)
# aggiungendo nuove feature basate sulle potenze di X fino al grado specificato (es. quadrato, cubo, ecc.).
poly = PolynomialFeatures(degree=4)
# Serve a preparare i dati: genera tutte le potenze di 𝑋 X che devono essere poi utilizzate nel modello di regressione.
X_poly = poly.fit_transform(X)
# Con Poly creiamo e addestriamo un altro modello LinearRegression
lin_reg_2 = LinearRegression()
# Serve ad addestrare il modello sulla base delle nuove feature (le potenze di 𝑋)
# e trovare i migliori coefficienti per effettuare previsioni.
lin_reg_2.fit(X_poly, y)

# Pyplot per LinearRegression(questo dovrebbe far vedere puntini e sbagliati)
pyplot_regressor(x=X, y=y, regressor=lin_reg, title='Giusto o Sbagliato', title_x='Level', title_y='Salary')
# Infatti se proviamo a predire il livello 6.5, notiamo che restituisce una somma che si aggira intorno ai 330K
# quando in realtà dovrebbe essere intorno ai 160 K
single_predict = lin_reg.predict([[6.5]])
print(single_predict)

# Pyplot per PolyLinearRegression
pyplot_regressor(x=X, X_poly=X_poly, y=y, regressor=lin_reg_2, title='Giusto o Sbagliato', title_x='Level',
                 title_y='Salary')
# Mentre con questo Poly, notiamo che la predizione del livello 6.5,
# si aggira intorno ai 159K. Molto più reale della LinearRegression
single_predict_poly = lin_reg_2.predict(poly.fit_transform([[6.5]]))
print(single_predict_poly)

"""
La Polynomial LinearRegression viene applicata quando i valori dipendenti(y) non solo lineari tra di loro, 
come se formassero una curva.Quindi quando una relazione tra valori indipendenti e valori dipendenti non è lineare.
Questa è la soluzione per determinati casi è la formula è :
y=β0+β1x+β2x2+β3x3+...+βnXn+ϵ

Dove:
𝑥2,𝑥3,...,𝑥𝑛  sono le potenze della variabile indipendente x.
𝛽0,𝛽1,...,𝛽𝑛β0,β1,...,βn sono i coefficienti della regressione.
n è il grado del polinomio (ad esempio,n=2 rappresenta un polinomio quadratico).

Quando usare la Polynomial Regression?
Usare la regressione polinomiale è utile quando:

La relazione tra la variabile dipendente e indipendente è non lineare: Ad esempio, se i dati seguono una curva 
(come una parabola), una regressione lineare semplice non sarà in grado di catturare adeguatamente la forma della relazione.
Si vuole ottenere un modello che possa adattarsi meglio ai dati con curve complesse.

Limiti della Polynomial Regression:

Overfitting: Se si aumenta troppo il grado del polinomio, il modello potrebbe adattarsi eccessivamente ai 
dati di addestramento, perdendo la capacità di generalizzare su nuovi dati.

Instabilità dei coefficienti: Aumentando il grado del polinomio, i coefficienti possono diventare
instabili e molto sensibili ai piccoli cambiamenti nei dati.
"""
