# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 17:05:27 2025

@author: trotta

Apprendimento non supervisionato
"""
# %% K-means -> E' per algoristmi non supervisionati come il CLUSTER (ovvero che non contengono etichette)
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers,cluster_std=blob_std, random_state=7)

k=5 # numero dei cluster
kmeans=KMeans(n_clusters=k,random_state=42)
y_pred=kmeans.fit_predict(X)
# %%
y_pred # restituisce le "etichette" di ogni cluster
# %%
kmeans.cluster_centers_ # retstituisce i centroidi (ovvero il loro punro di partenza da dove si iniziano a creare i cluster)
# %%
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]]) #->Nuove predizioni
kmeans.predict(X_new)
# %% Plotter per vedere come vengono divisi i cluster dove i centroidi avranno una "X"
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
# %% Si noteranno anche etichette al confine dei cluster che probabilmente sono sbagliati
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
save_fig("voronoi_plot")
plt.show()
# %%
kmeans.transform(X_new) # Restituisce la distanza di ogni istanza ad ogni centroide
# %% Metodi per iniziallizzare il centroide
good_init=np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])# Se sappiamo già dove sono posizionati i centroidi
kmeans=KMeans(n_clusters=5, init=good_init,n_init=1) # Impostare init con le loro posizioni e mettere n_init a 1 (questo evita di cercare ripetutamente i migliori centroidi)
#L'altra soluzione sarebbe quella di impostare n_init con un numero casuale che inizializzerà n volte per trovare i migliori centroidi (di default è impostato a 10)
# %% K-means ++ è la parte avanzata di k-means che consiste nel selezionare i centroidi distanti l'uno dall'altro : basta impostare "init="k-means++"" o altrimenti per il classico "random"
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=1, random_state=42)
kmeans.fit(X)
kmeans.inertia_
"""
In breve la differenza sta su come vengono inizializati i centroidi :
    K-means classico inizia raggruppandoli da vicino
    K-means++ li raggruppa da distanti l'uno dall'altro
"""
# %%







