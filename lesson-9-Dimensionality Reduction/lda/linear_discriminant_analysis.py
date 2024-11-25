import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

# LDA: nuova dimensionalità
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

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
LDA + un algoritmo SUPERVISIONATO simile al PCA ed è usato solamente per algoritmi di classificazioni.
"L'LDA funziona calcolando la media e la varianza di ciascuna classe e cercando le direzioni che massimizzano
la distanza tra le classi diverse, mantenendo la varianza minima all'interno delle stesse classi.
Questo processo è significativo solo quando si ha un problema con etichette discrete (come nel caso della classificazione),
perché senza le etichette di classe, l'LDA non saprebbe quali gruppi separare."
Esempio di Applicazione

L'LDA viene utilizzato in scenari come:
1 - Classificazione di immagini (es., riconoscimento facciale)
2 - Classificazione di documenti o testi
3 - Qualsiasi problema in cui le osservazioni sono assegnate a classi discrete e si desidera ridurre la dimensionalità
    mantenendo la capacità di separare queste classi.
"""
