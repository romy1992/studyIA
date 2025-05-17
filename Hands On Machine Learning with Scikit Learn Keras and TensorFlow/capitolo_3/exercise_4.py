# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:42:48 2025

@author: trotta

4.	Costruisci un classificatore di spam (un esercizio più impegnativo):
1.	Scarica esempi di spam e prosciutto dai set di dati pubblici di Apache SpamAssassin.
2.	Decomprimere i set di dati e acquisire familiarità con il formato dei dati.
3.	Suddividere i set di dati in un set di training e in un set di test.
4.	Scrivere una pipeline di preparazione dei dati per convertire ogni messaggio di posta elettronica in un vettore di funzionalità. La tua pipeline di preparazione dovrebbe trasformare un'e-mail in un vettore (sparso) che indica la presenza o l'assenza di ogni parola possibile. Ad esempio, se tutte le e-mail contengono solo quattro parole, "Ciao", "come", "sei", "tu", allora l'e-mail "Ciao Ciao Ciao tu" verrebbe convertita in un vettore [1, 0, 0, 1] (che significa ["Ciao" è presente, "come" è assente, "sono" è assente, "tu" è presente]), o [3, 0, 0, 2] se preferisci contare il numero di occorrenze di ogni parola.
Potresti voler aggiungere iperparametri alla tua pipeline di preparazione per controllare se eliminare o meno le intestazioni delle email, convertire ogni email in minuscolo, rimuovere la punteggiatura, sostituire tutti gli URL con "URL", sostituire tutti i numeri con "NUMBER" o persino eseguire lo stemming (ad esempio, tagliare le terminazioni delle parole; ci sono librerie Python disponibili per farlo).
Infine, prova diversi classificatori e verifica se riesci a creare un ottimo classificatore di posta indesiderata, con un elevato richiamo e un'elevata precisione.

"""

