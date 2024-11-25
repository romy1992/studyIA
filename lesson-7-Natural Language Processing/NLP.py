import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nltk.download('stopwords')

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""
NLP  è un campo dell'intelligenza artificiale e della linguistica computazionale che si occupa dell’interazione 
tra computer e linguaggio umano. L’obiettivo principale del NLP è consentire ai computer di comprendere, interpretare e 
generare linguaggio umano in modo utile e naturale.

Applicazioni Principali del NLP
Il NLP viene utilizzato in molte applicazioni pratiche, tra cui:

Analisi del Testo:
Estrazione di informazioni, sentiment analysis, estrazione di parole chiave, e analisi delle emozioni in testi 
come recensioni, articoli e post sui social media.

Chatbot e Assistenti Virtuali:
Assistenti come Siri, Alexa e Google Assistant usano NLP per comprendere le domande degli utenti 
e fornire risposte rilevanti.

Traduzione Automatica:
Servizi come Google Translate utilizzano tecniche di NLP per tradurre automaticamente testi da una lingua all’altra.

Riconoscimento e Sintesi Vocale:
NLP è usato per convertire la voce in testo (ASR, automatic speech recognition) e viceversa, 
come nei servizi di trascrizione e nelle sintesi vocali.

Raccomandazioni di Contenuti:
Sistemi di raccomandazione (come Netflix o Amazon) utilizzano NLP per suggerire contenuti rilevanti in base alla 
descrizione dei testi e alle preferenze dell'utente.

Analisi dei Documenti e Riassunti Automatici:
NLP è usato per sintetizzare e riassumere grandi quantità di testo, come articoli, documenti scientifici o 
report aziendali.
"""
