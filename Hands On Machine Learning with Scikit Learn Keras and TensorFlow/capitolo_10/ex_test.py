# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:09:27 2025

@author: trotta

DEEP Learning ANN
"""
# %%
import tensorflow as tf
from tensorflow import keras
fashion_mnist=keras.datasets.fashion_mnist # Dataset di abbigliamento
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data() # 28*28
# %%
X_valid,X_train=X_train_full[:5000] / 255.0,X_train_full[5000:] /255.0
y_valid,y_train=y_train_full[:5000],y_train_full[5000:]
# %%
import matplotlib.pyplot as plt
plt.imshow(X_train[0],cmap='binary')
plt.axis('off')
plt.show()
# %% Addestramento con il Dense 
import numpy as np
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))# Flatten : Converte l'array in 1D
model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
# %%
model.summary() # restituisce una tabella della tipologia di modello che sta per essere addestrato
model.layers[1] # possiamo anche prendere ogni layers e farci restitire le sue caratteristiche
keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
# %% Compilazione
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy']) # sgd -> addestramento con la discesa del gradiente stocatico
# Addestramento
history=model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=30) # validation_data -> è un set che aiuta a miglioraew il modello e ottimizzare i parametri.. Può essere sostituito con validation_spli=0.1 dove ad esempio prenderà solo il 10% dal modello di train 
# %%
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
# %%
model.evaluate(X_test,y_test)
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
# %% API SEQUENZUALE -> Sequantial come model e poi con il solito add o all'interno dell'istanza stessa
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing=fetch_california_housing()
X_train_full,X_test,y_train_full,y_test=train_test_split(housing.data, housing.target,random_state=42)
X_train,X_valid,y_train,y_valid=train_test_split(X_train_full,y_train_full,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_valid=sc.transform(X_valid)
X_test=sc.transform(X_test)

# %% API SEQUENZUALE
import tensorflow as tf
from tensorflow import keras
model=keras.models.Sequential([
    keras.layers.Dense(units=30,activation='relu',input_shape=X_train.shape[1:]),
    keras.layers.Dense(units=1)
    ])

model.compile(loss='mean_squared_error',optimizer=keras.optimizers.SGD(learning_rate=1e-3))
model.fit(X_train,y_train,epochs=20,validation_data=(X_valid,y_valid))
mse_test=model.evaluate(X_test,y_test)
# %% API FUNZIONALE -> a volte è consigliabile utilizzare questa per reti più complesse (anche se l'altra è più usata)
input_=keras.layers.Input(shape=X_train.shape[1:])# Si utilizza l'oggeto Input
hidden_1=keras.layers.Dense(units=30,activation='relu')(input_)
hidden_2=keras.layers.Dense(units=30,activation='relu')(hidden_1)
concat=keras.layers.concatenate([input_,hidden_2])
output=keras.layers.Dense(units=1)(concat)
model=keras.models.Model(inputs=[input_],outputs=[output])
"""
Perchè c'è a fine riga un (variabile)?
Questo viene messo a disposizione del modello (in questo caso Input) che al suo interno ha il metodo __call__()
Cosa fa __call__() ? -> applica la riga corrente alla variabile precedente : 
    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden_1 = keras.layers.Dense(units=30, activation='relu')(input_)
Ecco qui hidden_1 lo "aggiuge/trasforma" in input_
"""
# %%
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=20,validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
y_pred = model.predict(X_new)
# %% Per multi Input :
"""
Ci sono 2 input (A e B) ma solo B viene trasformato mentra A viene solo concatenato
Perchè questo?
Questo metodo serve per unire Input (quindi modelli) con diverse caratteristiche :
    A = Età, sesso, reddito, regione
    B = Embedding di testo o immagini
Infatti torneranno 2 output diversi per il tipo di "problema da predire"
"""
input_A = keras.layers.Input(shape=[5], name="wide_input")# piatto
input_B = keras.layers.Input(shape=[6], name="deep_input")# profondo
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20, validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))
# %% Per Multi Output :
"""
Come prima ma con doppio output finale
Da notare che il compile ha bisogno di una doppia loss perchè sono separati dando più peso all'output principale (0.9)
Caso d’uso	:
Output secondario può essere utile all’apprendimento	es. classificazione + regressione insieme
I dati del secondo output sono disponibili in training ma non in inference	es. diagnosi + biomarcatori
Regularizzazione implicita	il modello impara a non “overfitare” solo su un compito
"""
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.models.Model(inputs=[input_A, input_B],outputs=[output, aux_output])

model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))

total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
# %% Per il salvataggio del modello e la lettura
model.save("my_keras_model.h5")
model_save=keras.models.load_model("my_keras_model.h5")
# %% Callbacks -> e' un argmento che si può passare e keras utilizzera gli oggetti e li chiamerà all'inizio e alla fine dell'addestramento
# Per esempio possiamo chiamare un oggetto ModelCheckpoint che salverà il modello alla fine di ogni epoca che se utilizzato con l'argometo save_best_only=True , salverà il miglior modello dell'epoca

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])   

model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5.keras", save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,validation_data=(X_valid, y_valid),callbacks=[checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5.keras") # rollback to best model
mse_test = model.evaluate(X_test, y_test)
# %% EarlyStopping è un altro oggetto callbacks che bloccherà l'addestramento se entro un tot di epoche (patience) il modello non migliora.Quindi si bloccherà .

model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
mse_test = model.evaluate(X_test, y_test)
# %% Possiamo anche creare callbacks personalizzati , per esempio:
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))  

val_train_ratio_cb = PrintValTrainRatioCallback()
history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[val_train_ratio_cb])
"""
EXTRA -> possiamo utillizzarli anche tutti insieme : callbacks = [checkpoint_cb, early_stopping_cb, val_train_ratio_cb]
CHE POI VA INSERITO NEL FIT
"""
# %% 
import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir


"""
TensorBoard altro callbacks MOLTO UTILE

TensorBoard è un ottimo strumento di visualizzazione interattiva che puoi utilizzare per visualizzare le curve di apprendimento durante l'addestramento, 
confrontare le curve di apprendimento tra più esecuzioni, visualizzare il grafico di calcolo, analizzare le statistiche di addestramento, 
visualizzare le immagini generate dal tuo modello, visualizzare dati multidimensionali complessi proiettati in 3D e raggruppati automaticamente per te e altro ancora! 

Comandi per farlo partire in locale : 
load_ext tensorboard
tensorboard --logdir=./my_logs --port=6006
Ci sarà una dashboard con i diagrammi
"""

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])    
model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))



tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])
# %%
run_logdir2 = get_run_logdir()
run_logdir2
# %%
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])    
model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=0.05))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir2)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])
# %%
import numpy as np
help(keras.callbacks.TensorBoard.__init__)
test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(test_logdir) 
with writer.as_default():     
    for step in range(1, 1000 + 1):         
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)         
        data = (np.random.randn(100) + 2) * step / 100 # some random data         
        tf.summary.histogram("my_hist", data, buckets=50, step=step)         
        images = np.random.rand(2, 32, 32, 3) # random 32×32 RGB images         
        tf.summary.image("my_images", images * step / 1000, step=step)         
        texts = ["The step is " + str(step), "Its square is " + str(step**2)]
        tf.summary.text("my_text", texts, step=step)
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)         
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])         
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)

# %%
"""
Nel capitolo 10 leggere anche "Ottimizzazione degli iperparametri delle reti neurali" per trovare un giusto modo nel caso di bisogno

Parlerà anche del "Tasso di apprendimento" che è il più importante degli iperparametri e nell'ultimo esercizio (il 10) usa un algoritmo ExponentialLearningRate :
    Invece di mantenere fisso il learning rate, ExponentialLearningRate lo decresce (o talvolta cresce) esponenzialmente man mano che aumentano le epoche o gli step:
        Dove:
            decay è un fattore di decadimento esponenziale, tipicamente tra 0.9 e 0.99
            epoch è l'indice attuale dell'epoca di addestramento

        Perché è utile?
        All'inizio dell’addestramento si vuole un learning rate più alto per fare passi più grandi.
        Man mano che si avvicina al minimo della funzione di costo, si riduce il learning rate per stabilizzare la discesa ed evitare oscillazioni

"""
"""
Inoltre come facciamo a decidere quanti layer nascosti e nuroni usare?
Layer : La risposta è che solitamente si utilizzano un paio a meno che non siano reti complesse e bisogna quindi aumentare man mano finche non inizi a sovradattare
Meuroni : un pò come i layer,bisogno aumentare man mano finchè non inizia ad andare in overfitting
Solitamente si utilizzano sempre layer e neuroni in più per capire il dosaggio giusto dopo il primo test
"""