import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Leggo
df = pd.read_csv('../Wine.csv')

# Divido
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA: nuova dimensionalità
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Addestro con LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict
y_predict = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)
# Accuracy
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)

"""
PCA è un algoritmo NON SUPERVISIONATO che riduce la bidimensionalità dei dati per evitare che questi vadano in 
overfitting soprattutto per modelli che richiedono molte feature.
In questo esempio, se viene tolto il PCA , come accuracy restituirà 1, il che significa che questo è in overfitting
Parametri: n_components = 2 -> indica che la dimensionalità deve arrivare a 2 (Quindi bidimensionale)
fit_transform per dati di X_train
transform per dati di X_test
FARLO DOPO LO FUATURE SCALING E PRIMA DEL REALE FIT DI ADDESTRAMENTO(ESEMPIO LOGISTIREGRESSION) PER FAR SI CHE VENGA
APPLICATA LA NUOVA DIMENSIONALITA'
"""
