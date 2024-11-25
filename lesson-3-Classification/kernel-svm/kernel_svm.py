import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Leggo
df = pd.read_csv('../Social_Network_Ads.csv')

# Divido X e y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Divido in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Addestro
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Previsione sui dati di test NON visti
y_prediction = classifier.predict(X_test)

# Esempio di previsione singola
print(classifier.predict(sc.transform([[30, 87000]])))

# Confusion Matrix
cm = confusion_matrix(y_test, y_prediction)
print(cm)
# Accuracy
accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)


"""
Il kernel SVM (rbf), rispetto a neighbors e vector_machine(lineare), è il più adatto per situazioni non lineari.
In altri casi lineari, dove bastava tracciare una linea obliqua dove a sinistra finivano i negativi e a destra i 
positivi, RBF traccia una linea non lineare che serve per rendere al meglio determinate accuratezze.

Parametri principali del Kernel SVM:
kernel: Specifica il tipo di kernel da utilizzare. I più comuni sono:
'linear': Per separazione lineare.
'rbf': Per relazioni non lineari.
'poly': Kernel polinomiale per problemi non lineari con forme più complesse.
'sigmoid': Simile alla funzione attivazione di una rete neurale.
"""