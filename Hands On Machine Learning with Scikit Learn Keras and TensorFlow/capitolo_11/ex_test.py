# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 09:47:12 2025

@author: trotta

Addestramento di reti neurali profonde : 
    In questo capito si spiega come possono esserci problemi durante l'addestramento e delle funzioni che possiamo inserire per evitare ciò.
    Si parla del problema del gradiente che può esplodere/nascondersi:
        - Nascondersi : A causa della retropropagazione(backpropagation), verso i livelli inferiori i neuroni iniziano a non funzionare più, quindi 
                       può capitare che metà dei 100 neuroni non funzioni più e quindi l'allenamento non converge mai verso una buona soluzione.
        - Espodere : Il contrario, i gradienti possono diventare sempre più grandi tanto da divergere l'algoritmo
La soluzione a ciò possono essere le funzioni di attivazioni (nel capitolo spiega i relu e tutti i suoi sottoinsieme) ma soprattutto esiste un normalizzatore
detto "Batch Normalization"(BN) in grado di riequilibrare il tutto.
"Metafora BN" : 
    Un personal Trainer allena un gruppo di 20 ragazzi dove :
        - 10 sono iperattivi 
        - 10 sono deboli
    Cosa fa quindi il personal Trainer? Cerca di riequilibrare gli esercizi in modo tale che entrambi i gruppi arrivino allo stesso livello
Sintomi : 
    Non esiste una regola assoluta, ma ci sono sintomi chiari che ti fanno sospettare che inserire la Batch Normalization (BN) possa aiutare. Vediamoli:
🚨 5 segnali che indicano che BN può aiutarti:
1. 📉 Loss che scende molto lentamente o a scatti
Durante l'addestramento, la loss fatica a scendere o ha una discesa instabile (a zig-zag).

Sintomo che le attivazioni interne stanno cambiando troppo (internal covariate shift).

🔧 Inserire BN stabilizza l’apprendimento e rende la discesa più fluida.

2. 🔁 Overfitting troppo rapido
Accuracy altissima sul training, ma bassissima sul validation/test.

BN ha un effetto regolarizzante, simile al dropout, che può aiutare a contenere l’overfitting.

3. ⚠️ Gradienti instabili
Problemi di exploding o vanishing gradient: gradienti troppo grandi o troppo piccoli.

Rete che si “blocca” (loss non cambia) o esplode (loss va a nan).

🧯 BN tiene sotto controllo le attivazioni e previene questi problemi.

4. 🧠 La rete ha molti layer profondi
Se usi una rete molto profonda (es. >10 layer), BN è quasi sempre utile.

Aiuta i gradienti a fluire meglio all’indietro, migliorando la retropropagazione.

5. 🚫 Learning rate troppo sensibile
Cambi di learning rate piccoli causano grandi differenze di prestazioni.

BN rende la rete più robusta a valori di learning rate più alti o non perfetti.


"""
# %% Batch Normalization
from tensorflow import keras

model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),# Riequilibria per ogni strato il normalizzatore
    keras.layers.Dense(300,activation='relu'),
    keras.layers.BatchNormalization(),# Riequilibria per ogni strato il normalizzatore
    keras.layers.Dense(100,activation='relu'),
    keras.layers.BatchNormalization(),# Riequilibria per ogni strato il normalizzatore
    keras.layers.Dense(10,activation='softmax')
    ])

model.summary()
# %%
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10,validation_data=[X_valid,y_valid])
# %% Ritaglio sfumatura(Gradient Clipping) : altra tipologia per "normalizzare" è ritagliare e sfumare i gradienti durante la retropropagazione
# basta aggiungere nell'optimizer : 
optimizer = keras.optimizers.SGD(clipvalue=1.0)
optimizer = keras.optimizers.SGD(clipnorm=1.0)
# entrambi sono validi
# %% TASSO DI APPRENDIMENTO "learning_rate"
"""
Molto importante per gli ottimizzatori :
    Se troppo alto all'inizio farà progressi molto rapidamente, ma finirà per ballare intorno all'ottimale, senza mai stabilizzarsi veramente
    Se troppo basso convergerà verso l'ottimale, ma ci vorrà molto tempo
Quindi esistono delle tecniche per trovare il giusto tasso e implementarlo anche durante l'addestramento:
"""
# Decay statico (decresce nel tempo) : La decay fa scendere lr ad ogni step secondo
optimizer = keras.optimizers.Adam(learning_rate=0.01, decay=1e-4)

# Callback dinamica con ReduceLROnPlateau : Riduce il learning rate se la validazione smette di migliorare
from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)
model.fit(X_train,y_train, epochs=20, validation_split=0.1, callbacks=[lr_scheduler])

# Custom scheduler o LearningRateScheduler : riduci del 10% ogni epoca dopo la 5
from tensorflow.keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * 0.9  # riduci del 10% ogni epoca dopo la 5
lr_schedule = LearningRateScheduler(scheduler)
model.fit(X_train,y_train, epochs=20, callbacks=[lr_schedule])
# %%

""" EVITARE L'OFERFITTING -> TECNICA DROPOUT GIA' USATA : ATTIVA E SPEGNE I NEURONI PER EVITARE SOVRADDATTAMENTI -> LA PIU' USATA
| Tecnica       | Tipo             | Applicazione            | Scopo Principale                     | Usata con             |
| ------------- | ---------------- | ----------------------- | ------------------------------------ | --------------------- |
| Dropout       | Regolarizzazione | Training                | Ridurre overfitting                  | Qualsiasi attivazione |
| Alpha Dropout | Regolarizzazione | Training (con SELU)     | Mantiene self-normalization con SELU | Solo con SELU         |
| MC Dropout    | Incertezza       | Training + Test         | Stima dell’incertezza del modello    | Task bayesiani        |
| Max-Norm      | Vincolo sui pesi | Dopo ogni aggiornamento | Limitare la crescita dei pesi        | Qualsiasi rete        |

"""