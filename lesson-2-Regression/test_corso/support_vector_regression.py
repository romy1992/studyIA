import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Leggo il csv
df = pd.read_csv('Data.csv')

# Divido in X e y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Controllo se ci sono NaN o encode da fare (non per questo esempio)

# Divido in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Qui, a differenza degli altri modelli, devo applicare il feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Addestro con i modelli appena scalati
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

# Previsione
y_predict = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1, 1))

score = r2_score(y_test, y_predict)
print(score)
