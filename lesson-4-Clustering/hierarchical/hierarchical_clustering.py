import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch  # ->Libreria usata per disegnare il dendrogramma
from sklearn.cluster import AgglomerativeClustering

# Leggo
df = pd.read_csv('../Mall_Customers.csv')

# Recupero i dati X
X = df.iloc[:, [3, 4]].values

"""
Dendrogramma:
Rappresenta come i cluster si fondono o si dividono e serve per capire quanti Cluster servono che a differenza di 
k-means++ si sapeva già.
Come leggere in dendrogramma?
Bisogna misurare la misura verticale dell'immagine e trovare quella più alta per poi contare i punti di intersecazione.
Info : https://www.udemy.com/course/machinelearning/learn/lecture/5714428#questions
"""
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))  # ward->riduce la varianza dentro il dendrogramma
plt.title('Dendrogram')
plt.xlabel('Clienti')
plt.ylabel('Euclidean distances')
plt.show()

# Addestro con predict con i 3 o 5(entrambi andavano bene nelle misure) cluster trovati nel dendrogramma precedente
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc = hc.fit_predict(X)
print(y_hc)

# Per visualizzare i 5 cluster con il pyplot
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(0, 5):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=colors[i], label=f'Cluster {i + 1}')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
"""
Hierarchical Clustering(Utilizzato sempre per questioni marketing)
è un metodo di apprendimento non supervisionato(COME K-MEANS++) utilizzato per raggruppare i dati in cluster
basati sulla loro somiglianza, creando una gerarchia di gruppi.

A differenza di K-means, non richiede di specificare in anticipo il numero di cluster, 
e il risultato è una struttura ad albero chiamata dendrogramma.

Differenza con k-means++?
In sintesi, K-means è più adatto per problemi di clustering con dati grandi e cluster ben definiti, mentre Hierarchical 
clustering è utile quando si vuole esplorare la struttura dei dati in modo più approfondito, 
senza predefinire il numero di cluster.
"""
