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
# %%
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()

fetch_spam_data()
# %%