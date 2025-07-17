# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:50:27 2025

@author: trotta
13.	Addestra un modello utilizzando un ciclo di addestramento personalizzato per affrontare il set di dati Fashion MNIST (vedi Capitolo 10).
1.	Visualizza l'epoca, l'iterazione, la perdita media di addestramento e l'accuratezza media in ogni epoca (aggiornata a ogni iterazione), 
    nonché la perdita di convalida e l'accuratezza alla fine di ogni epoca.
2.	Prova a utilizzare un ottimizzatore diverso con una velocità di apprendimento diversa per i livelli superiori e inferiori.

"""
from tensorflow import keras

        