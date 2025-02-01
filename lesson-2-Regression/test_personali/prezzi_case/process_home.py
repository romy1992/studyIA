import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

model = joblib.load('best_model.pkl')

# Leggo il dataset
dataset = pd.read_csv('prezzi_case.csv')

# X e y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaler x
sc = model.named_steps['scaler']
nuovi_dati_scaled = sc.transform(X_test)

# Predizione
type_model = model.named_steps['model']
y_predict = type_model.predict(nuovi_dati_scaled)
print(y_predict)

# Mse
mse = mean_squared_error(y_test, y_predict)
print(f'Mean squared error : {round(mse, 2)}')

# R2_score
r2 = r2_score(y_test, y_predict)
print(f'R2 score : {round(r2, 2)}')

X_test_p = X_test[:, 0]
plt.scatter(X_test_p, y_test, color='red')
plt.show()
