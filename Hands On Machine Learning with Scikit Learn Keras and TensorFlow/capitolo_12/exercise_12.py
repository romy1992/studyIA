# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:38:45 2025

@author: trotta

12.	Implementa un livello personalizzato che esegua la normalizzazione dei livelli (useremo questo tipo di livello nel Capitolo 15):
1.	Il  metodo build() dovrebbe definire due pesi addestrabili α e β, entrambi di forma input_shape[-1:] 
    e tipo di dati tf.float32. α deve essere inizializzato con 1 e β con 0.
2.	Il  metodo call() dovrebbe calcolare la μ media e la  deviazione standard σ delle caratteristiche di ciascuna istanza. 
    Per questo, puoi usare tf.nn.moments(inputs, axes=-1, keepdims=True), che restituisce la media μ e la varianza σ2 di tutte le istanze
    (calcola la radice quadrata della varianza per ottenere la deviazione standard).
    Quindi la funzione dovrebbe calcolare e restituire α⊗(X - μ)/(σ + ε) + β, dove ⊗ rappresenta la moltiplicazione itemwise (*) e ε è un termine di livellamento (piccola costante per evitare la divisione per zero, ad esempio 0,001).
3.	Assicurati che il tuo livello personalizzato produca lo stesso output (o quasi lo stesso) 
    del livello keras.layers.LayerNormalization.

"""
from tensorflow import keras
import tensorflow as tf

class LayerNormalization(keras.layers.Layer):
    def __init__(self,eps=0.001,**kwargs):
        super().__init__(**kwargs)
        self.eps=eps
    
    # Il  metodo build() dovrebbe definire due pesi addestrabili α e β, entrambi di forma input_shape[-1:] 
    # e tipo di dati tf.float32. α deve essere inizializzato con 1 e β con 0.
    def build(self,batch_input_shape):
        self.alpha = self.add_weight(name="alpha", shape=batch_input_shape[-1:],initializer="ones")
        self.beta = self.add_weight(name="beta", shape=batch_input_shape[-1],initializer="zeros")
        super().build(batch_input_shape) # must be at the end
      
    # Il  metodo call() dovrebbe calcolare la μ media e la  deviazione standard σ delle caratteristiche di ciascuna istanza. 
    def call(self,X):
        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / (tf.sqrt(variance + self.eps)) + self.beta
    
    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "eps": self.eps}
