# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:58:46 2025

@author: trotta

2.	Scrivi una funzione in grado di spostare un'immagine MNIST in qualsiasi direzione (sinistra, destra, su o giù) di un pixel.  
Quindi, per ogni immagine nel set di training, creare quattro copie spostate (una per direzione) e aggiungerle al set di training. 
Infine, addestra il tuo modello migliore su questo set di addestramento esteso e misurane l'accuratezza sul set di test. 
Dovresti notare che il tuo modello funziona ancora meglio ora! 
Questa tecnica di crescita artificiale del set di addestramento è chiamata aumento dei dati o espansione del set di addestramento.
"""
# %%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,as_frame=False)
X, y = mnist["data"], mnist["target"]
# %%
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def shift_image(image, dx, dy):# Metodo che sposta l'immagine
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

def pyplot(title,image):
    plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.title(title, fontsize=14)
    plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")

X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=42)
image=X_train[1001]
shifted_image_down=shift_image(image, 0, 5)
shifted_image_left=shift_image(image, -5, 0)

plt.figure(figsize=(12,3))
pyplot("Originale", image)
pyplot("Sotto", shifted_image_down)
pyplot("Sinistra", shifted_image_left)
plt.show()
# %% Creo e aggiungo al dataset la copia di 4 immagini spostate     
import numpy as np
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)
# %% mescolare (shuffle) un dataset in modo casuale mantenendo il corretto allineamento tra features (X) e target (y).
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn_cls=KNeighborsClassifier(n_neighbors=4,weights='distance')
knn_cls.fit(X_train_augmented, y_train_augmented)
y_pred=knn_cls.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)




