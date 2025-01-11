import numpy as np
import pandas as pd
from matplotlib.pyplot import bone, pcolor, colorbar, plot, show
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Import
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Scaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

"""
Training SOM
x=10, y=10:
La SOM è una griglia 10x10 di nodi (neuronali). Ogni nodo rappresenta un vettore prototipo che si adatta ai dati.

input_len=15: 
Ogni vettore di input ha 15 dimensioni.

sigma=1.0: 
È il raggio della funzione di vicinanza (Gaussian Neighborhood). 
Determina quanto influisce l'aggiornamento del nodo vincitore sui nodi vicini nella griglia.

learning_rate=0.5: 
Velocità di apprendimento. Controlla quanto velocemente i nodi si adattano agli input.
"""
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
"""
I pesi iniziali dei nodi della SOM vengono impostati casualmente, utilizzando come riferimento il dataset 
X (un array numpy con N campioni e 15 caratteristiche).
"""
som.random_weights_init(X)

"""
data=X: Passi il dataset X alla SOM.

num_iteration=100: La SOM sarà aggiornata in 100 iterazioni. Durante ogni iterazione:
- Un campione casuale x del dataset X viene scelto.

- Si individua il Best Matching Unit (BMU), cioè il nodo il cui peso è il più vicino al campione 
  x (misurato con una distanza, come la distanza euclidea).

- I pesi del nodo BMU e dei suoi vicini nella griglia vengono aggiornati in modo da avvicinarsi al campione x, con 
  un'intensità controllata da: sigma (vicinanza tra nodi) learning_rate (adattamento nel tempo).
"""
som.train_random(data=X, num_iteration=100)

# Pyplot
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Cerco la frode
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
frauds = sc.inverse_transform(frauds)
print(frauds)

"""
SOM : Self Organization Maps (Mappe organizzate), è un algoritmo di DL NON SUPERVISIONATO.
E' una rete neurale che riduce dimensionalità e organizza i dati in modo che simili vengano
mappati vicini in uno spazio bidimensionale.
"""
