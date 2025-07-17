# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 16:03:21 2025

@author: trotta
"""
# %% 
"""
API dei dati : sono strumenti che permettono di elaborare, caricare e fornire dati ai modelli sia durante il training che durante l'inferenza
*Inferenza : tutto ciò che accade dopo il training (prevedere,aggioanre ecc)

A cosa servono?
- Gestire grandi quantità di dati.
- Ottimizzare il caricamento in memoria.
- Applicare trasformazioni sui dati (normalizzazione, augmentazione, ecc.).
- Eseguire operazioni in parallelo e pipeline efficienti.grazie

"""
import tensorflow as tf

X=tf.range(10)
dataset=tf.data.Dataset.from_tensor_slices(X)
print(X)
print(dataset)
for item in dataset:
    print(item)
    
# Equivale a :
dataset = tf.data.Dataset.range(10)
for item in dataset:
    print(item)
# %% repate(3) -> riprodurrà il set per 3 volte
dataset=dataset.repeat(3).batch(7, drop_remainder=True) # batch(7)-> numeri all'interno dell'array
for item in dataset:
    print(item) # solitamente ci sarà sempre(quasi) una parte finale con residui numeri (in qeusto caso tf.Tensor([8 9], shape=(2,), dtype=int64))
    # Per liminare questi residui : basta inserire "drop_remainder=True"
# %%
dataset = dataset.map(lambda x: x * 2) # map()usato per trasformare il dato (per esempio se si vuole ruotare l'immagine)
for item in dataset:
    print(item)
# %%
#dataset = dataset.apply(tf.data.experimental.unbatch()) # Now deprecated
#dataset = dataset.unbatch()
dataset = dataset.filter(lambda x: x < 10)  # filter : prende solo ementi < 10
for item in dataset.take(3):
    print(item)
# %%
dataset=tf.data.Dataset.range(10).repeat(3)
dataset=dataset.shuffle(buffer_size=3,seed=42).batch(7)# shuffle : rimescola i dati all'interno del dataset
for item in dataset:
    print(item)
"""
Se chiami repeat() su un set di dati mescolato, per impostazione predefinita genererà un nuovo ordine ad ogni iterazione. 
Questa è generalmente una buona idea, ma se si preferisce riutilizzare lo stesso ordine ad ogni iterazione 
(ad esempio, per i test o il debug), è possibile impostare reshuffle_each_iteration=False.
"""
# %%
"""
TFRecord : E' un formato binario
    
Perché usare TFRecord?
È compatto, quindi occupa meno spazio rispetto ai file CSV o JSON.
È veloce da leggere, soprattutto per dataset molto grandi.
Si integra perfettamente con le pipeline di tf.data.
Ottimizzato per letture sequenziali e distribuite.

Come funziona?
Salvi dati come una sequenza di record serializzati.
Ogni record può contenere immagini, testo, label, feature numeriche, ecc.
"""

"""
🔥 In sintesi:
Le API di dati (tf.data) ti aiutano a creare pipeline efficienti per leggere e trasformare dati.
I TFRecord sono un formato binario ideale per salvare dataset grandi e strutturati, da usare in combinazione con le API di dati.


TFRecord: COSA È E A COSA SERVE
🔸 È un formato di memorizzazione dati, pensato per salvare dataset in modo binario, compatto ed efficiente.
🔸 Serve quando hai dataset molto grandi, come milioni di immagini, audio o dati tabellari.
🔸 È indipendente da TensorFlow nel senso che è solo un formato file.
🔸 Permette:
🚀 Lettura sequenziale veloce.
🚀 Accesso efficiente da disco o da storage distribuito (come Google Cloud o AWS).
🚀 Minor uso di RAM e I/O rispetto a CSV, immagini sparse o JSON.
🟨 ➝ TFRecord è un contenitore di dati.
✅ tf.data: COSA È E A COSA SERVE
🔸 È un'API di pipeline dati.
🔸 Serve per:
🛠️ Caricare dati da sorgenti (file CSV, immagini, TFRecord, ecc.).
🧠 Preprocessare i dati (resize, normalizzazione, augmentazione...).
🚚 Gestire il batch, lo shuffle, il prefetch (caricamento anticipato dei dati per accelerare il training).
🔸 Funziona su qualunque sorgente dati: file, array, generatori Python e anche su TFRecord.
🟦 ➝ tf.data è un sistema per gestire il flusso di dati durante il training.


 Metafora semplice:
🔸 TFRecord è la dispensa: organizza e conserva gli ingredienti (i dati) in modo ordinato e compatto.
🔸 tf.data è la cucina: prende gli ingredienti, li trasforma, li porziona e li serve pronti al modello per essere "mangiati" (addestrati).

"""
# %%
import tensorflow as tf
ocean_prox_vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
ocean_proximity = tf.feature_column.categorical_column_with_vocabulary_list("ocean_proximity", ocean_prox_vocab)

ocean_proximity_one_hot = tf.feature_column.indicator_column(ocean_proximity)
ocean_proximity_embed = tf.feature_column.embedding_column(ocean_proximity,dimension=2)
"""
| Tipo                        | Forma del Vettore | Esempio                              | Quando usarlo                      |
| --------------------------- | ----------------- | ------------------------------------ | ---------------------------------- |
| **One-Hot (`indicator`)**   | `[0, 1, 0, 0, 0]` | Categoria esplicita                  | Poche categorie, semplice          |
| **Embedding (`embedding`)** | `[0.13, -0.27]`   | Rappresentazione appresa dal modello | Tante categorie, relazioni latenti |
"""