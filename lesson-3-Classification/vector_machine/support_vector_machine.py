import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Leggo il csv
df = pd.read_csv('../Social_Network_Ads.csv')

# Divido in X e y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Divido il modello
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Faccio il Future Scaling per i dati di train perché hanno valori sballati tra di loro
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Addestro
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predict single
print(classifier.predict(sc.transform([[30, 87000]])))

# Predict
y_predict = classifier.predict(X_test)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_predict)
print(cm)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)
