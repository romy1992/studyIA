import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Leggo i dataframe
df = pd.read_csv('../Social_Network_Ads.csv')

# Divido in X e y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling con i dati di ingresso
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Addestro
classifier = DecisionTreeClassifier(random_state=0, criterion='entropy')
classifier.fit(X_train, y_train)

# Predict con i dati di test
y_predict = classifier.predict(X_test)
print(y_predict)

# Predict con dati singoli
print(classifier.predict(sc.transform([[30, 87000]])))

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Accuracy
score = accuracy_score(y_test, y_predict)
print(score)

"""
Utilizza un singolo albero decisionale per classificare i dati. Questo lo rende interpretabile,
ma può essere facilmente soggetto a overfitting, 
cioè adattarsi troppo ai dati di training e generalizzare male su nuovi dati.
"""