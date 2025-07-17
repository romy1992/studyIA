# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 16:29:31 2025

@author: trotta
"""
# %%
from tensorflow import keras
import numpy as np

(X_train_full,y_train_full),(X_test,y_test) = keras.datasets.fashion_mnist.load_data()
X_valid,X_train=X_train_full[:-5000],X_train_full[-5000:]
y_valid,y_train=y_train_full[:-5000],y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# %%
from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME") # E' un alias per evitare ripetizioni

"""
- Filters sono come una piccola finestra che passa man mano sull'immagine per analizzarla e prendere le caratteristiche (es.64) 
che produrrà 64map dove ogni caratteristica prenderà una un bordo, curva , pattern ecc.Più filtri ci saranno più la rete impara


- MaxPooling serve per ridurre la dimensione spazioale(altezza e larghezza) delle feature map dopo i layer convoluzionali :
    prima :
        [[1, 3, 2, 4],[5, 6, 1, 2], [8, 7, 3, 1],[4, 2, 9, 0]]
    
    dopo :
        [[6, 4],[8, 9]]
    Oltre ad essere più efficiente , riduce anche l'overfitting

- Il Flatten DOPO tutto il processo serve per appiattire il 1D perchè i layer convoluzionali e pooling producono strati 3D (altezza,larghezza e canali) ma il Dense accetta solo vettori:
    prima : [batch_size, 3, 3, 128]
    dopo : [batch_size, 3 * 3 * 128] = [batch_size, 1152]

E da li in poi di lavora in maniera classica
"""
model = keras.models.Sequential([
    DefaultConv2D(filters=64,kernel_size=7,input_shape=[28,28,1]), # I filtri aumenteranno man mano nella CNN(pratica comune)
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax')
   ])
# %%
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] # pretend we have new images
y_pred = model.predict(X_new)
