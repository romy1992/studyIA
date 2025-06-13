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
# %%

