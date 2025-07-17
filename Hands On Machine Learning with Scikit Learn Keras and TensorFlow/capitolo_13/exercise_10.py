# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 12:11:51 2025

@author: trotta

10.	In questo esercizio scaricherai un set di dati, lo dividerai, creerai un tf.data.Dataset per caricarlo e preelaborarlo in modo efficiente, 
quindi costruirai e addestrerai un modello di classificazione binaria contenente un  livello di incorporamento:
1.	Scarica il Large Movie Review Dataset, che contiene 50.000 recensioni di film dall'Internet Movie Database. 
    I dati sono organizzati in due directory, train e test, ciascuna contenente una  sottodirectory pos con 12.500 recensioni positive e una  sottodirectory neg con 12.500 recensioni negative. 
    Ogni recensione viene memorizzata in un file di testo separato. Ci sono altri file e cartelle (incluso il sacchetto di parole pre-elaborato), ma li ignoreremo in questo esercizio.
2.	Suddividere il set di test in un set di convalida (15.000) e un set di test (10.000).
3.	Usa tf.data per creare un set di dati efficiente per ogni set.
4.	Creare un modello di classificazione binaria, usando un  livello TextVectorization per pre-elaborare ogni revisione. Se il  livello TextVectorization non è ancora disponibile (o se ti piacciono le sfide), prova a creare il tuo livello di pre-elaborazione personalizzato: puoi utilizzare le funzioni nel  pacchetto tf.strings, ad esempio lower() per rendere tutto minuscolo, regex_replace() per sostituire la punteggiatura con gli spazi e split() per dividere le parole sugli spazi. È consigliabile utilizzare una tabella di ricerca per generare indici di parole, che devono essere preparati con il  metodo adapt().
5.	Aggiungi un  livello di incorporamento e calcola l'incorporamento medio per ogni recensione, moltiplicato per la radice quadrata del numero di parole (vedi Capitolo 16). Questo incorporamento medio riscalato può quindi essere passato al resto del modello.
6.	Addestra il modello e verifica la precisione ottenuta. Cerca di ottimizzare le tue pipeline per rendere la formazione il più veloce possibile.
7.	Utilizzare TFDS per caricare più facilmente lo stesso set di dati: tfds.load("imdb_reviews").


"""

