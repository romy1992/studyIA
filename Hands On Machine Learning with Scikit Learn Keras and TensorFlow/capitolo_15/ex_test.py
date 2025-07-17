# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:40:47 2025

@author: trotta

RNN
"""
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
# %% Questo metodo crea la base per la creazione dei vettoriali RNN in 3D
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
# %% 
"""
Prima di iniziare a utilizzare le RNN, è spesso una buona idea avere alcune metriche di base, 
altrimenti potremmo finire per pensare che il nostro modello funzioni alla grande quando in realtà sta facendo peggio dei modelli di base. 
Ad esempio, l'approccio più semplice consiste nel prevedere l'ultimo valore di ogni serie. 
Questa si chiama previsione ingenua e a volte è sorprendentemente difficile sovraperformare. 
""" 
from keras.losses import MeanSquaredError
y_pred = X_valid[:, -1]
mse = MeanSquaredError()
np.mean(mse(y_valid, y_pred))
# %% Addestramento
model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50,1]),
    keras.layers.Dense(1)
    ])
model.compile(loss="mse",optimizer="adam")
history=model.fit(X_train,y_train, validation_data=(X_valid,y_valid),epochs=20)
# %%
model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)
print(y_pred)
# %% Simple RNN
"""
Esempio :
    Se stai cercando di prevedere il valore successivo in una serie temporale, tipo:
    [1.1, 1.2, 1.3, 1.2, 1.5, 1.4, 1.6] → predici 1.7
    ✅ RNN capisce la dinamica temporale → migliore per la previsione.
    ❌ Dense+Flatten ignora l'ordine → meno efficace, anche se può funzionare in problemi semplici o lineari.
"""
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
])
optimizer = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
# %%
model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)
# %% Deep RNN
"""
Questo SimpleRNN con più impilazioni, è per architetture più profonde :
    - Il primo strato ha 20 neuroni , con return_sequences=True per tutti i livelli ricorrenti 
        (tranne l'ultimo, se ti interessa solo l'ultimo output). 
        In caso contrario, verranno emessi un array 2D (contenente solo l'output dell'ultimo passaggio temporale) 
        anziché un array 3D (contenente output per tutti i passaggi temporali) 
        e il livello ricorrente successivo si lamenterà del fatto che non gli stai alimentando le sequenze nel formato 3D previsto.
    - Il secondo strato segue il primo
    - Il terzo ha solo un neurone per l'output finale
"""
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(1)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# %%
model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)
# %%
"""
Qui fa vedere la stessa cosa ma con il Dense finale dove :
    Più flessibile
    Più comune in pratica (soprattutto con LSTM/GRU)
    Più facile da estendere (es. se vuoi output multivariato: Dense(3), ecc.)
"""
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
# %%
model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)

"""
| Modello                            | Considera ordine? | Memoria temporale? | Profondità | Output finale       |
| ---------------------------------- | ----------------- | ------------------ | ---------- | ------------------- |
| `SimpleRNN(1)`                     | ✅                 | ✅                  | 1 livello  | ultimo timestep     |
| `Flatten + Dense`                 | ❌                 | ❌                  | 1 livello  | regressione lineare |
| `3 x SimpleRNN`                   | ✅                 | ✅✅                 | 3 livelli  | ultimo timestep     |
| `2 x SimpleRNN + Dense`           | ✅                 | ✅✅                 | 3 livelli  | Dense(1) su output RNN finale |

"""
# %% Possiamo anche precedere più passi temporali avanti e non solo uno , ad esempio 10
"""
In questo modo si prevedono 10 passi temporali uno alla volta
"""
series = generate_time_series(1, n_steps + 10) # Crea i 10 passi temporali in più
X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)

Y_pred = X[:, n_steps:]
# %%
"""
Questa versione invece prevere i 10 passi contemporaneamente
"""
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]
# %%
np.mean(mse(Y_valid, Y_pred))
# %%
Y_naive_pred = np.tile(X_valid[:, -1], 10) # take the last time step value, and repeat it 10 times
np.mean(mse(Y_valid, Y_naive_pred))
# %% Con versione Dense + Flatten (Ci impiega meno tempo)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(10)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))
# %% Con versione SimpleRNN + Dense (Ci impiega più tempo)
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))
# %%
series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, -10:, :]
Y_pred = model.predict(X_new)[..., np.newaxis]
# %%
"""
Ora invece prevediamo i 10 step successivi contemporaneamente ma a sqeunza a sequenza
"""
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]
# %%
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

def last_time_step_mse(Y_true, Y_pred): # Infatti questo metodo iniettato nel metrics compile, restiuisce ogni step mse
    return mse(Y_true[:, -1], Y_pred[:, -1])

model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))
# %%
series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

