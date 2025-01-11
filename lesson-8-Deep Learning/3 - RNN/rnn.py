import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Parte 1 - Data Preprocessing

# Leggo il dataset di train
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# Prendo i dati di ingresso
training_set = dataset_train.iloc[:, 1:2].values

# Applico il feature scaling ma con la classe di MinMaxScaler che è consigliata in RNN
sc = MinMaxScaler(feature_range=(0, 1))  # feature_range: indica il range di scaler (di default è già 0 e 1)
training_set_scaled = sc.fit_transform(training_set)

# Creo la struttura con 60 time steps e un output
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i - 60:i, 0])  # recupera le righe dei 60 giorni precedenti dell'unica colonna(0)
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

"""
Ridimensiono ->
np.reshape :
    Serve a cambiare la forma di un array. Puoi specificare il numero di dimensioni e come devono essere distribuiti gli elementi.
    Non modifica i dati interni dell'array, ma solo la loro disposizione.

shape[...]:
    Serve per accedere alle dimensioni dell'array. Non cambia l'array, ma restituisce un valore o una tupla con le dimensioni attuali.

Quando si lavora con modelli di deep learning (es. RNN, LSTM, CNN), spesso i dati devono essere in una forma specifica:
RNN/LSTM: Generalmente vogliono dati di forma (samples, timesteps, features).
Aggiungendo una dimensione extra (es. (X_train.shape[0], X_train.shape[1], 1)), si indica che ogni campione ha una sola feature per timestep.
"""
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Parte 2 - Building RNN
"""
Aggiunta di un layer LSTM : (Long Short-Term Memory) sono un tipo di rete ricorrente (RNN) progettata per apprendere dipendenze temporali a lungo termine, particolarmente utile per dati sequenziali o temporali.
units : sono i neuroni
return_sequences=True:
    Specifica se il layer deve restituire l'intera sequenza di output (un valore per ogni timestep) o solo l'ultimo valore della sequenza.
    Quando True, il layer restituisce una sequenza 3D (batch, timesteps, features).
    
input_shape=(X_train.shape[1], 1):
Definisce la forma dell'input per il layer:
X_train.shape[1]: Numero di timesteps (lunghezza della sequenza temporale per ogni campione).
1: Numero di feature (dimensioni per ogni timestep, ad esempio un solo valore per timestep).
Esempio: Se hai una sequenza temporale con 60 timestep e 1 feature, la forma sarà (60, 1)
"""
regressor = Sequential()
# 1 Strato di LSTM con input_shape
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))  # Per evitare l'overfitting, userà il 20% dei neuroni inseriti prima
# 2 Strato di LSTM senza input_shape perché già dichiarato
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# 3 Strato di LSTM senza input_shape perché già dichiarato
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# 4 Strato di LSTM senza return_sequences perché è l'ultimo passaggio di output
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# Aggiungere l'ultimo layer du Output Dense
regressor.add(Dense(units=1))

# Compile
regressor.compile(loss='mean_squared_error', optimizer='adam')

# Fit
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Parte 3 - Fare previsione e visualizzare il risultato

# Prendere le reali vendite del 2017 di Google
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
# Concatenare i 2 dataset in maniera verticale con axis=0 (quindi uno sotto l'altro)
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# Recuperare i dati di 60 giorni precedenti al 01/03/2017 (quindi i 2 mesi finali del 2016)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# Ridimensionare
inputs = inputs.reshape(-1, 1)
# Applicare il transform dello scaling perché in fase di addestramento è stato applicato lo scaling
inputs = sc.transform(inputs)
# Creo la struttura con 60 time steps e un output
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i - 60:i, 0])  # recupera le righe dei 60 giorni precedenti dell'unica colonna(0)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Faccio la previsione
predict_stock_price = regressor.predict(X_test)
# Inverto il transform prima creato
predict_stock_price = sc.inverse_transform(predict_stock_price)

# Metrics
rmse = math.sqrt(mean_squared_error(real_stock_price, predict_stock_price))
print(rmse)

# Visualizzare il risultato
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predict_stock_price, color='Blue', label='Predict Google Stock Price')
plt.title('Google Stock Price Predict')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
"""
UTILE SE I DATASET HANNO SEQUENZE TEMPORALI PER FAR SI CHE L'ALGORITMO "RICORDI" EVENTI PASSATI
E' SEQUENZIALE COME ALGORITMO(QUINDI SI BASA SU EVENTI TEMPORALI)

Le RNN (Recurrent Neural Networks) sono un tipo di rete neurale progettate per elaborare dati sequenziali, 
come serie temporali, testo o audio.La caratteristica principale delle RNN è che hanno connessioni ricorrenti 
che consentono di mantenere una "memoria" degli stati precedenti. 
Questo le rende ideali per analizzare informazioni dipendenti dal contesto, come il significato di una parola in una frase.

Come funzionano:
Ogni neurone riceve sia l'input corrente che il suo stato precedente, aggiornando così una rappresentazione interna (memoria).
Sono calcolate ricorsivamente, passo dopo passo lungo la sequenza.

Limiti:
Vanishing gradient: difficoltà nell'apprendere dipendenze a lungo termine.
LSTM (Long Short-Term Memory) e GRU (Gated Recurrent Units) sono versioni migliorate delle RNN, capaci di gestire meglio
le dipendenze lunghe.

Esempi d'uso:
Traduzione automatica.
Generazione di testo.
Predizione di serie temporali (es. stock market).

In sintesi, le RNN permettono ai modelli di "ricordare" informazioni per analizzare sequenze nel tempo!
"""
