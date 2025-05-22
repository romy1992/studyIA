# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:52:47 2025

@author: trotta

"""
# %%
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X=iris["data"][:,(2,3)]
y=(iris["target"]==2).astype(np.float64)#.astype(np.float64)-> converte il booleano in 0 o 1
# SVM LINEAR CLASSIFICATION
"""
Iperparametro :
    C -> PENSIAMO A 2 CAREGGIATE DOVE TROVIAMO UNA LINEA CHE DIVIDE I DATI(CHE SEPARA I 2 VERSI DELLA CAREGGIATA).
        POI TROVIAMO I BORDI DI ENTRAMBE LE CAREGGIATE TRATTEGGIATE , ECCO :
            PIù "C" è BASSO, PIù LE LINEE TRATTEGGIATE SI ALLONTANANO DALLA LINEA CONTINUA A META'
            PIù "C" P ALTO, PIù LE LINEE VANNO VERSO LA LINEA CONTINIUA
            QUANDO UNA CLASSE SI TROVA DENTRO QUESTA CLASSE , VIENE DETTA QUINDI VIOLAZIONE DEL MARGINE E DI SOLITO
            E' MEGLIO AVERNE POCHI MA A SUA VOLTA , SE CI SONO PIù VIOLAZIONI, GENERALIZZERA' MEGLIO
            SE IL MODELLO è IN OVERFITTING, REGOLANDO IL PARAMETRO "C" SI PUò SISTEMARE
    LOSS "HINGE"-> 
        
""" 
svm_clf=Pipeline([
    ('scaler' , StandardScaler()),
    ('linear_svc' , LinearSVC(C=1,loss='hinge',random_state=42))])
svm_clf.fit(X,y)
# %%
svm_clf.predict([[5.5,1.7]])
# %% 
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

# SVM NON-LINEAR CLASSIFICATION
"""
La differenza è nel creare una Pipeline con un TRASFORMATORE polinomiale tipo "PolynomialFeatures"
"""
X,y=make_moons(n_samples=100, noise=0.15,random_state=42)#-> make_moons : è un dataset artificiale
polinomial_svm=Pipeline([
    ('poly_features',PolynomialFeatures(degree=3)),# Trasformatore
    ('scaler',StandardScaler()),#Scalatore
    ('svm_clf',LinearSVC(C=1,loss='hinge',random_state=42))])#Modello lineare
# Tutta questa pipeline crea un SVM NON-LINEAR

polinomial_svm.fit(X,y)
polinomial_svm.predict([[5.5,1.7]])
# %%
# POLYNOMIAL KERNEL CLASSIFICATION
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "svm"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
    
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


polynomial_kernel=Pipeline([
    ('scaler',StandardScaler()),
    ('svm_clf',SVC(kernel='poly',C=5,degree=3, coef0=1))])

polynomial_kernel.fit(X,y)

polynomial100_kernel=Pipeline([
    ('scaler',StandardScaler()),
    ('svm_clf',SVC(kernel='poly',C=5,degree=10, coef0=100))])

polynomial100_kernel.fit(X,y)
"""
I 2 SVC hanno il kernel a poly (quindi è polinomiale) dove il primo con polinomio di 3 e il secondo con il polinomio di 10
Più e alto e più il modello è complesso.
Il coef0 invece più è alto , più il modello è "rigido" e tende a favorire decisioni più complesse (potenzialmente overfitting).
"""

fig, axes = plt.subplots(ncols=2, figsize=(10.5, 4), sharey=True)

plt.sca(axes[0])
plot_predictions(polynomial_kernel, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.sca(axes[1])
plot_predictions(polynomial100_kernel, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)
plt.ylabel("")

save_fig("moons_kernelized_polynomial_svc_plot")
plt.show()
# %%
# Gaussian RBF Kernel CLASSIFICATION
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
"""
kernel="rbf" -> è più flessibile

"gamma" -> indica dove è localizzata la decisione: 
    - più è grande più è localizzata nei punti di training  
    - più è piccolo e più larga e liscia
"""
rbf_kernel_svm_clf.fit(X, y)
# %%
from sklearn.svm import SVC

gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)

for i, svm_clf in enumerate(svm_clfs):
    plt.sca(axes[i // 2, i % 2])
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
    if i in (0, 1):
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")

save_fig("moons_rbf_svc_plot")
plt.show()
# %%
# REGERESSIONE SVM
"""
LA DIFFERENZA CON CLASSIFICATION (CHE CERCA DI LIMITARE LE VIOLAZIONI DEI MARGINI) :

SVM per Classificazione ->SPIEGATO PRIMA
Obiettivo: trovare il margine massimo tra due classi.
Immagina che i tuoi dati siano due gruppi di punti (classe 0 e classe 1). L'SVM cerca una retta (o superficie) che:

divide i due gruppi,
massimizza la distanza (il "margine") tra la retta e i punti di ciascuna classe,
penalizza i punti che cadono all'interno del margine o sul lato sbagliato.

SVM per Regressione (SVR)
Obiettivo: trovare una curva (o retta) che:

passa più vicino possibile ai punti,
ma lascia una fascia di tolleranza dentro cui non importa se i punti sono esattamente toccati.

Questa fascia si chiama "epsilon-tube" (ε-tubo), ed è una zona di non penalizzazione:
    
Se il punto è dentro il tubo (cioè l’errore è minore di ε), non viene penalizzato.
Se il punto è fuori dal tubo (errore maggiore di ε), allora si applica una penalità proporzionale all'errore fuori margine.

Esempio pratico: ε in azione
Con ε = 1.5: la fascia è larga, quindi il modello non cerca di toccare tutti i punti. È più tollerante agli errori.
Con ε = 0.5: la fascia è stretta, quindi il modello cerca di stare molto vicino ai dati, aumentando il rischio di overfitting.
"""
np.random.seed(42)
m = 50
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5) 
svm_reg.fit(X, y)
# %%

svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)
svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)
svm_reg1.fit(X, y)
svm_reg2.fit(X, y)

def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
    return np.argwhere(off_margin)

svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)

eps_x1 = 1
eps_y_pred = svm_reg1.predict([[eps_x1]])
# %%
def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)

fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])
plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)
plt.ylabel(r"$y$", fontsize=18, rotation=0)
#plt.plot([eps_x1, eps_x1], [eps_y_pred, eps_y_pred - svm_reg1.epsilon], "k-", linewidth=2)
plt.annotate(
        '', xy=(eps_x1, eps_y_pred), xycoords='data',
        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
    )
plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
plt.sca(axes[1])
plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)
save_fig("svm_regression_plot")
plt.show()
# %%


