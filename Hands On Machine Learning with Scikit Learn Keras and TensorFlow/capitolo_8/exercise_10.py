# -*- coding: utf-8 -*-
"""
Created on Wed May 28 17:31:51 2025

@author: trotta

10.	
Utilizza t-SNE per ridurre il set di dati MNIST a due dimensioni e traccia il risultato utilizzando Matplotlib. 
È possibile utilizzare un grafico a dispersione utilizzando 10 colori diversi per rappresentare la classe di destinazione di ogni immagine. 

In alternativa, è possibile sostituire ogni punto nel grafico a dispersione con la classe dell'istanza corrispondente (una cifra da 0 a 9),
o anche tracciare versioni ridotte delle immagini delle cifre stesse (se si tracciano tutte le cifre, la visualizzazione sarà troppo ingombrante, 
quindi è necessario disegnare un campione casuale o tracciare un'istanza solo se nessun'altra istanza è già stata tracciata a distanza ravvicinata). 

Dovresti ottenere una bella visualizzazione con gruppi di cifre ben separati. 

Prova a utilizzare altri algoritmi di riduzione della dimensionalità come PCA, LLE o MDS e confronta le visualizzazioni risultanti.
"""
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

mnist = fetch_openml('mnist_784', version=1,as_frame=False)
m = 10000
idx = np.random.permutation(60000)[:m]
X = mnist['data'][idx]
y = mnist['target'][idx]
# %%
tsne=TSNE(n_components=2,random_state=42)
X_tsne=tsne.fit_transform(X)
# %%
plt.figure(figsize=(13,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()
