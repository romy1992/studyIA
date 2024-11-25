import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

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
classifier = GaussianNB()
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
E' basato sul Teorema di Bayes e assume che tutte le feature siano indipendenti tra loro. È semplice, veloce,
e funziona bene con piccoli dataset e quando le feature sono debolmente correlate. 
Tuttavia, l'assunzione di indipendenza è spesso irrealistica e può ridurre la precisione se le feature sono correlate.
"""