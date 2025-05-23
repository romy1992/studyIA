import pandas as pd
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

# Addestro con il LinearRegression (In questo caso Multi perché i valori indipendenti sono multipli)
regression = LinearRegression()
regression.fit(X_train, y_train)

# Faccio la predizione su quelli di test perché sono valori che lui non conosce
y_predict = regression.predict(X_test)

# Lo score più è quasi vicino all' 1 più è attendibile
# I valori da prendere sono quelli di y_test e quelli della predizione precedente
score = r2_score(y_test, y_predict)
print(score)
