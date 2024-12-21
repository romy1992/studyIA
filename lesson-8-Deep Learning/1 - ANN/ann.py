import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# Leggo dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Suddivido in X e y
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# LabelEncoder per trasformare in 0 e 1 i valori di (Maschio e Femmina)
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# OneHotEncoder
cm = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(cm.fit_transform(X))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# FeatureScaling obbligatorio per le reti neurali
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
Creo i layer :
"ann.add"-> Definisce i layer del modello. Ogni layer aggiunge una trasformazione lineare (detta anche "peso") 
            seguita da una funzione di attivazione, che introduce non-linearità.
"units"-> Rappresenta il numero di neuroni nel layer. 
         Ogni neurone calcola una somma pesata degli input seguita dalla funzione di attivazione.
"activation" -> La "funzione di attivazione" decide come trasformare l'output dei neuroni. Tra le più comuni troviamo :
    relu : (Rectified Linear Unit): Trasforma gli input negativi in 0, mentre lascia invariati i valori positivi.
            È semplice ed efficace, molto usata nei layer nascosti.
    sigmoid : Mappa i valori in ingresso a un intervallo tra 0 e 1, rendendola utile per problemi di 
                classificazione binaria. Tuttavia, soffre del problema della scomparsa del gradiente quando gli input 
                sono molto grandi o molto piccoli (flat saturation).
    tanh : (Tangente iperbolica): Simile alla sigmoidea, ma con un intervallo tra -1 e 1. 
            Risolve parzialmente il problema della saturazione rendendo la funzione centrata su 0. 
            Tuttavia, soffre ancora della scomparsa del gradiente.
    softmax : Un'estensione multiclasse della sigmoidea. Trasforma i valori in probabilità 
            (che sommano a 1) per problemi di classificazione con più categorie.
            Ad esempio, dato un set di immagini, può restituire le probabilità per ogni classe di oggetti.
    
SPIEGAZIONE PER CUI L'ULTIMO LAYER E' DIVERSO DAGLI ALTRI:
    L'ultimo layer è diverso perchè si deve adattare alla risoluzione del problema.
    Infatti dovrà avere 1 neurone è come funzione di attivazione potrà essere :
        sigmoid: Per output di classificazione binario (0 o 1)
        softmax: Per multi output
        None: Per output di regressione
"""
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""
Dopo aver definito il modello, bisogna compilarlo:
loss:
È la funzione di perdita utilizzata per quantificare quanto il modello sta "sbagliando" durante l'addestramento.
La scelta della funzione di perdita dipende dal tipo di problema:
    binary_crossentropy: Per problemi di classificazione binaria, misura la distanza tra le probabilità predette (sigmoid) e i valori target.
    categorical_crossentropy: Per classificazione multiclasse con softmax e target codificati one-hot.
    sparse_categorical_crossentropy: Simile a categorical_crossentropy, ma i target sono interi (non one-hot).
    mean_squared_error: Per problemi di regressione, misura la differenza media al quadrato tra predizioni e target.

optimizer:
    È l'algoritmo che aggiorna i pesi della rete neurale per minimizzare la funzione di perdita.
    Gli optimizer più comuni:
    adam: Uno dei più popolari, combina i vantaggi di RMSprop e SGD, adattandosi dinamicamente al tasso di apprendimento.
    sgd (Stochastic Gradient Descent): Aggiorna i pesi gradualmente, utile in problemi semplici o per il fine-tuning.
    rmsprop: Ottimizza i problemi con gradienti non uniformi.
    adagrad e adadelta: Adattano il tasso di apprendimento in base alla frequenza di aggiornamento dei parametri.

metrics:
    Specifica quali metriche monitorare durante l'addestramento. Non influisce sull'addestramento, ma aiuta a valutare il modello.
    Le metriche più comuni:
    accuracy: Percentuale di predizioni corrette rispetto ai target.
    precision e recall: Usate in classificazione per misurare la qualità delle predizioni positive.
    mean_squared_error o mae (Mean Absolute Error): Usate nei problemi di regressione.
"""
ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""
batch_size:
    Specifica il numero di campioni elaborati prima di aggiornare i pesi del modello.
    Esempio:
    Se hai 1000 campioni e batch_size=32, il modello elaborerà i dati in gruppi da 32 campioni per volta, aggiornando i pesi dopo ogni batch.
    Vantaggi:
    Batch più piccoli possono velocizzare l'addestramento, ma potrebbero produrre gradienti più "rumorosi".
    Batch più grandi richiedono più memoria e possono rallentare l'addestramento, ma producono gradienti più stabili.
    Default: Se non specificato, il batch size è 32.
    
epochs:
    Numero di volte in cui l'intero set di dati viene passato attraverso la rete neurale durante l'addestramento.
    Esempio:
    Se hai 1000 campioni e epochs=100, il modello vedrà quei 1000 campioni 100 volte.

verbose (opzionale):
    Controlla il livello di dettaglio mostrato durante l'addestramento.
    0: Nessuna uscita.
    1: Mostra una barra di avanzamento per ogni epoca.
    2: Mostra solo un riepilogo per ogni epoca.
"""
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Predizione singola
y_predict_single = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print(f'Single predict : {y_predict_single > 0.5}')

# Predizione test
y_predict = ann.predict(X_test)
y_predict = (y_predict > 0.5)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)
