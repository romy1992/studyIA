import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Leggo il csv
df = pd.read_csv('Data.csv')

# Divido in X e y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Controllo se ci sono NaN o encode da fare (non per questo esempio)

# Divido in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Prima applichiamo il Modello di Poly
poly_reg = PolynomialFeatures(degree=4)
# Applico la formula ai dati di Train(dati non visti)
X_poly = poly_reg.fit_transform(X_train)
# Addestro con il LinearRegression
regression = LinearRegression()
regression.fit(X_poly, y_train)

# Faccio la predizione
y_pre = regression.predict(poly_reg.transform(X_test))

score = r2_score(y_test, y_pre)
print(score)
