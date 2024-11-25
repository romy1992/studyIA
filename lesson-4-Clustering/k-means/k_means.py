import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Leggo
df = pd.read_csv('../Mall_Customers.csv')

# A differenza degli altri test, la y(valori dipendenti) non ci sono
X = df.iloc[:, [3, 4]].values  # ->Qui gli sto dicendo di prendermi la colonna 3 e 4

# Non si utilizza lo spit proprio perché non esiste la variabile dipendete

"""
Addestriamo :
K-means(Aggiornato con k-means++) è un algoritmo di apprendimento NON SUPERVISIONATO a differenza del solito X e y dove
si sanno già le y (variabili dipendenti) che sono algoritmi di apprendimento SUPERVISIONATO.

Quando conviene usarlo?
In questo esempio lo utilizza per capire quali clienti 
hanno bisogno di maggior "pubblicità" e quindi per questioni di marketing

OPPURE :

Segmentazione del mercato:
Dividere i clienti in gruppi omogenei in base ai loro comportamenti di acquisto, 
preferenze o caratteristiche demografiche, per creare strategie di marketing mirate.

Analisi esplorativa dei dati:
Identificare pattern o raggruppamenti naturali nei dati per una migliore comprensione e 
analisi preliminare dei dataset.

Compressione di immagini:
Ridurre la quantità di colori in un'immagine raggruppandoli in cluster simili, 
mantenendo la qualità visiva (ad esempio, per ridurre la memoria occupata da un'immagine).

Riconoscimento di pattern:
Identificare gruppi simili all'interno di dataset complessi, come immagini,
segnali audio o dati comportamentali.

Clustering di documenti:
Raggruppare documenti, articoli o post sui social media in base alla somiglianza dei contenuti, 
utile per categorizzare testi o per suggerimenti automatici.

Anomalie nei dati:
Identificare valori anomali o punti fuori norma, poiché i dati che non appartengono a nessun cluster possono indicare
anomalie o outlier.
"""
wcss = []
for i in range(1, 11):
    # Per ogni Cluster inizializziamo il k-means++ che è un'evoluzione di k-means
    kmeans = KMeans(random_state=42, n_clusters=i, init='k-means++')
    # Addestro
    kmeans.fit(X)
    # Aggiungo nella lista di wcss i k-means addestrati
    wcss.append(kmeans.inertia_)
"""
Pyplot ti farà visualizzare i giusti Cluster da utilizzare.
Come capirlo?
Nell'esempio con il pyplot sotto, bisogna analizzare il diagramma e capire la retta dove inizia la sua fase di stabilità
in questo esempio è 5 .

WCSS (Within-Cluster Sum of Squares) è una metrica utilizzata nell'algoritmo di K-means 
clustering per misurare la qualità della suddivisione dei dati in cluster. 
In particolare, rappresenta la somma dei quadrati delle distanze 
tra ciascun punto dati e il centroide del proprio cluster.

Significato del WCSS:
Il WCSS misura la dispersione interna ai cluster: 
quanto i punti dati all'interno di un cluster sono vicini al centroide di quel cluster.

Un valore WCSS più basso indica che i punti dati sono più vicini ai centroidi, 
quindi i cluster sono più compatti e ben definiti.
"""
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Una volta quindi capito in quanti cluster devi utilizzare, bisogna addestrarlo
kmeans = KMeans(n_clusters=5, random_state=42, init='k-means++')
# Eseguo quindi l'addestramento e la predizione
y_kmeans = kmeans.fit_predict(X)

# Per visualizzare i 5 cluster con il pyplot
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(0, 5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i + 1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

"""
In sintesi, K-means è consigliato quando si vuole esplorare i dati, 
identificare gruppi nascosti o ridurre la dimensionalità senza supervisione.
"""
