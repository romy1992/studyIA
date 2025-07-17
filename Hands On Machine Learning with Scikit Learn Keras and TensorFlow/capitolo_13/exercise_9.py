# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 12:11:50 2025

@author: trotta

9.	Caricare il set di dati Fashion MNIST (introdotto nel Capitolo 10), dividerlo in un set di addestramento, 
    un set di convalida e un set di test, mescolare il set di addestramento e salvare ogni set di dati in più file TFRecord. 
    Ogni record dovrebbe essere un prototipo di esempio serializzato con due funzionalità: 
    l'immagine serializzata (utilizzare tf.io.serialize_tensor() per serializzare ogni immagine) e l'etichetta.  
Quindi usa tf.data per creare un set di dati efficiente per ogni set. 
Infine, utilizza un modello Keras per addestrare questi set di dati, incluso un livello di pre-elaborazione per standardizzare ogni funzionalità di input. 
Provare a rendere la pipeline di input il più efficiente possibile, utilizzando TensorBoard per visualizzare i dati di profilatura.

"""
from tensorflow import keras
import tensorflow as tf
# Caricare il set di dati Fashion MNIST (introdotto nel Capitolo 10)
(X_train_full,y_train_full),(X_test,y_test) = keras.datasets.fashion_mnist.load_data()
X_valid,X_train = X_train_full[:5000],y_train_full[5000:]
y_valid,y_train = y_train_full[:5000],y_train_full[5000:]
# %%
# Dividerlo in un set di addestramento, un set di convalida e un set di test e mescolare il set di addestramento e salvare ogni set di dati in più file TFRecord
train_set = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(len(X_train))
valid_set = tf.data.Dataset.from_tensor_slices((X_valid,y_valid))
test_set = tf.data.Dataset.from_tensor_slices((X_test,y_test))
# %% L'immagine serializzata (utilizzare tf.io.serialize_tensor() per serializzare ogni immagine) e l'etichetta.  
def create_example(image,label):
    image_data=tf.io.serialize_tensor(image)
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data.numpy()])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }))
for image, label in valid_set.take(1):
    print(create_example(image, label))
# %%
