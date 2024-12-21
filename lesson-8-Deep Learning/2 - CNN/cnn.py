import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

# Preprocessing training set
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalizza i pixel dell'immagine tra 0 e 1
    shear_range=0.2,  # Applica una trasformazione di taglio (shear) fino al 20%
    zoom_range=0.2,  # Applica uno zoom casuale fino al 20%
    horizontal_flip=True  # Inverte orizzontalmente alcune immagini
)
train_set = train_datagen.flow_from_directory(
    './training_set',  # Percorso della cartella contenente le immagini di training
    target_size=(64, 64),  # Ridimensiona tutte le immagini a 64x64 pixel
    batch_size=32,  # Numero di immagini per batch
    class_mode='binary'  # Le etichette delle immagini sono binarie (es. Gatto/Cane)
)

# Preprocessing test set
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = train_datagen.flow_from_directory('./test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

"""
Il processo di Convolutional consiste in questi passaggi : 
Convolution: Estrazione delle feature (bordi, texture).
Pooling: Riduzione della dimensione spaziale (es. max pooling).
Flattening: Conversione in un vettore 1D.
Full Connection: Combinazione delle feature per la classificazione.
Output Layer: Previsione finale (es. probabilità per ogni classe).
"""
cnn = tf.keras.models.Sequential()

# Convolution
"""
filters=32:
Indica il numero di filtri (o kernel) utilizzati in questo livello di convoluzione.
Ogni filtro apprende una caratteristica specifica (es. bordi, angoli, texture) dalla matrice di input.
Più filtri = più caratteristiche estratte, ma anche maggiore complessità computazionale.

kernel_size=3:
La dimensione del filtro o kernel. In questo caso, il kernel è una matrice 3x3.
Durante la convoluzione, il kernel scorre attraverso l'immagine e calcola i prodotti puntuali per generare una mappa delle caratteristiche.
I kernel più piccoli (es. 3x3) sono comuni perché catturano dettagli locali senza aumentare troppo la complessità.

activation='relu':
La funzione di attivazione applicata dopo la convoluzione.
ReLU (Rectified Linear Unit) elimina i valori negativi, impostandoli a 0, e lascia invariati i valori positivi.
Introduce non-linearità nel modello, permettendogli di apprendere relazioni complesse.

input_shape=[64, 64, 3]:
Specifica la forma del dato di input solo per il primo livello della rete.
64, 64: Dimensioni dell'immagine (altezza x larghezza).
3: Numero di canali dell'immagine (es. 3 per immagini RGB a colori).
Le immagini vengono ridimensionate a 64x64 pixel prima di essere elaborate.
"""
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Pooling
"""
pool_size=2:
Definisce la dimensione della finestra del pooling (2x2).
La finestra scorre sull'immagine e riduce ogni regione a un singolo valore (es. il massimo, nel caso del max pooling).
Una finestra 2x2 dimezza la dimensione spaziale dell'immagine (es. da 64x64 a 32x32).

strides=2:
Indica il numero di pixel di spostamento della finestra di pooling a ogni passo.
Se strides=2, la finestra si sposta di 2 pixel alla volta, evitando sovrapposizioni.
strides=2 è spesso utilizzato con pool_size=2 per garantire un downsampling regolare (dimezzare le dimensioni).
"""
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Added a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattering
cnn.add(tf.keras.layers.Flatten())

# Full Connection: con 128 neuroni e relu come funzione consigliabile prima dell'output finale
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer: con l'ultimo neurone di risposta
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Addestro
cnn.fit(x=train_set, validation_data=test_set, epochs=25)

# Predict
test_image = image.load_img('./single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(train_set.class_indices)
prediction = 'Cane' if result[0][0] == 1 else 'Gatto'
print(prediction)
