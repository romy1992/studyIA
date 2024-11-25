import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Leggo
dataframe = pd.read_csv('../Social_Network_Ads.csv')

# Divido in X e y
X = dataframe.iloc[:, [0, 1]].values
y = dataframe.iloc[:, -1].values

# Controllo se c'è da pulire qualcosa

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Addestro
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predict
y_predict = classifier.predict(X_test)

# Matrix Confusion
cm = confusion_matrix(y_test, y_predict)
print(f'Matrix Confusion :\n {cm}')

# Accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy : {accuracy}')

# Cross Val Score (k-fold)
accuracies = cross_val_score(X=X_train, y=y_train, cv=10, estimator=classifier)
print(f'Accuracies mean : {accuracies.mean()}')
print(f'Accuracies std : {accuracies.std()}')

"""
Il k-fold viene applicato per capire se l'accuratezza e il bias-variance è nella metrica giusta
Nei parametri, oltre ad aggiungere i modelli di train e il modello di fit, dobbiamo aggiungere il parametro cv.

Se imposti cv a un numero intero, ad esempio cv=10, cross_val_score esegue una k-fold cross validation con il numero di
fold indicato, in questo caso, 10.
Quindi, suddivide il dataset in 10 parti e per ogni iterazione usa un fold come set di validazione e gli altri 9 come 
set di addestramento, ripetendo per tutti i fold.

In questo caso , ogni fold avrà la sua metrica.Infatti il mean() fa la media di tutti i fold effettuati.
Mentre il metodo std() restituisce la differenza della media,ovvero quel valore che si aggira tra 
il mean()-std() e mean()+std, esempio :
Questo mean() restituisce un 90% e std un 6%.
Questo significa che 90-6=84 e 90+6=96 è la soglia a cui si aggirano i k-fold.
"""
