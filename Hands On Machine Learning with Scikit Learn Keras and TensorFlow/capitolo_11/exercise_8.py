# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:01:57 2025

@author: trotta
8.	Esercitati ad addestrare una rete neurale profonda sul set di dati di immagini CIFAR10:
    
1.	Costruisci una DNN con 20 strati nascosti di 100 neuroni ciascuno (è troppo, ma è il punto di questo esercizio). Utilizzare l'inizializzazione He e la funzione di attivazione ELU.
2.	Utilizzando l'ottimizzazione Nadam e l'arresto anticipato, addestra la rete sul set di dati CIFAR10. Puoi caricarlo con keras.datasets.cifar10.load_ data(). Il set di dati è composto da 60.000 immagini a colori da 32 × 32 pixel (50.000 per l'addestramento, 10.000 per i test) con 10 classi, quindi avrai bisogno di un livello di output softmax con 10 neuroni. Ricordarsi di cercare la giusta velocità di apprendimento ogni volta che si modifica l'architettura o gli iperparametri del modello.
3.	Ora prova ad aggiungere la normalizzazione batch e confronta le curve di apprendimento: sta convergendo più velocemente di prima? Produce un modello migliore? In che modo influisce sulla velocità di allenamento?
4.	Provare a sostituire la normalizzazione batch con SELU e apportare le modifiche necessarie per garantire che la rete si auto-normalizzi (ad esempio, standardizzare le funzionalità di input, utilizzare l'inizializzazione normale LeCun, assicurarsi che la DNN contenga solo una sequenza di strati densi, ecc.).
5.	Provare a regolarizzare il modello con l'interruzione alfa. Quindi, senza ripetere l'addestramento del modello, verifica se è possibile ottenere una migliore precisione utilizzando MC Dropout.
6.	Riaddestra il tuo modello utilizzando la pianificazione 1cycle e verifica se migliora la velocità di addestramento e l'accuratezza del modello.

"""
# %% 
from tensorflow import keras
# %%
"""
1.	Costruisci una DNN con 20 strati nascosti di 100 neuroni ciascuno. Utilizzare l'inizializzazione He e la funzione di attivazione ELU.
"""
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
# %%
"""
2.	Utilizzando l'ottimizzazione Nadam e l'arresto anticipato, addestra la rete sul set di dati CIFAR10. 
Puoi caricarlo con keras.datasets.cifar10.load_data(). 
Il set di dati è composto da 60.000 immagini a colori da 32 × 32 pixel (50.000 per l'addestramento, 10.000 per i test) con 10 classi, quindi avrai bisogno di un livello di output softmax con 10 neuroni.                                                                                                                                                                                                                                         
Ricordarsi di cercare la giusta velocità di apprendimento ogni volta che si modifica l'architettura o gli iperparametri del modello.
"""
dataset=keras.datasets.cifar10.load_data()
model.add(keras.layers.Dense(10, activation='softmax'))

optimizer=keras.optimizers.Nadam(learning_rate=5e-5)
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=optimizer)


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]


early_stopping_cb = keras.callbacks.EarlyStopping(patience=20) #Arresto anticipato
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5.keras", save_best_only=True)# salva il modello migliore

import os
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "my_cifar10_logs", "run_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)# crea in remoto dei grafici per ogni epoca
callbacks=[early_stopping_cb,checkpoint_cb,tensorboard_cb]

model.fit(X_train,y_train,epochs=100,validation_data=(X_valid,y_valid),callbacks=callbacks)

# %%
model=keras.models.load_model("my_keras_model.h5.keras")
model.evaluate(X_valid,y_valid)

# %%
"""
3. Ora prova ad aggiungere la normalizzazione batch e confronta le curve di apprendimento: 
sta convergendo più velocemente di prima? 
Produce un modello migliore? 
In che modo influisce sulla velocità di allenamento?
"""
model_bn=keras.models.Sequential()
model_bn.add(keras.layers.Flatten(input_shape=(32,32,3)))
model_bn.add(keras.layers.BatchNormalization())
for _ in range(20):
    model_bn.add(keras.layers.Dense(100,kernel_initializer='he_normal'))
    model_bn.add(keras.layers.BatchNormalization())
    model_bn.add(keras.layers.Activation("elu"))
model_bn.add(keras.layers.Dense(10, activation='softmax'))
optimizer=keras.optimizers.Nadam(learning_rate=5e-5)
model_bn.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=optimizer)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20) #Arresto anticipato
checkpoint_cb = keras.callbacks.ModelCheckpoint("model_bn.keras", save_best_only=True)# salva il modello migliore

run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "my_cifar10_logs", "run_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)# crea in remoto dei grafici per ogni epoca
callbacks=[early_stopping_cb,checkpoint_cb,tensorboard_cb]

model_bn.fit(X_train,y_train,epochs=100,validation_data=(X_valid,y_valid),callbacks=callbacks)
# %%
model_bn=keras.models.load_model("model_bn.keras")
model_bn.evaluate(X_valid,y_valid)
# %%
"""
4.Provare a sostituire la normalizzazione batch con SELU e apportare le modifiche necessarie per garantire che la rete si auto-normalizzi 
(ad esempio, standardizzare le funzionalità di input, utilizzare l'inizializzazione normale LeCun, 
assicurarsi che la DNN contenga solo una sequenza di strati densi, ecc.).
"""
# %%
"""
5.Provare a regolarizzare il modello con l'interruzione alfa. 
Quindi, senza ripetere l'addestramento del modello, verifica se è possibile ottenere una migliore precisione utilizzando MC Dropout.
"""
# %%
"""
6.Riaddestra il tuo modello utilizzando la pianificazione 1cycle e verifica se migliora la velocità di addestramento e l'accuratezza del modello.
"""


