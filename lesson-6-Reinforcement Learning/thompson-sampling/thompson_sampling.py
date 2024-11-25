import random

import matplotlib.pyplot as plt
import pandas as pd

# Leggo
dataframe = pd.read_csv('../Ads_CTR_Optimisation.csv')

# Creo il giro per l'addestramento
N = 10000  # Righe del dataset
d = 10  # Colonne del dataset
ads_selected = []  # Lista annunci selezionati
numbers_rewards_1 = [0] * d  # Lista di ricompensa 1
numbers_rewards_0 = [0] * d  # Lista di ricompensa 0
total_rewards = 0
for n in range(0, N):
    ad = 0
    max_reward = 0
    for i in range(0, d):
        # Applico la formula di Thompson : B(NI1(n)+1,NI0(n)+1)
        random_beta = random.betavariate(numbers_rewards_1[i] + 1, numbers_rewards_0[i] + 1)
        if random_beta > max_reward:
            max_reward = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataframe.values[n, ad]
    if reward == 1:
        numbers_rewards_1[ad] += 1
    elif reward == 0:
        numbers_rewards_0[ad] += 1
    total_rewards += reward

# Istogramma con pyplot
plt.hist(ads_selected)
plt.title('Istogramma degli annunci selezionati')
plt.xlabel('Annunci')
plt.ylabel("Numeri dell'annuncio selezionato")
plt.show()

"""
L'algoritmo di Thompson è un algoritmo PROBABILISTICO(A differenza di quello UBC che è Deterministico).
E' un algoritmo che fà parte di quelli del REINFORCEMENT LEARNING , ed è simili a quello dell'UBC , ovvero che sceglie
un'azione casuale per massimizzare la ricompensa attesa.
"Probabilistico, basato su campionamento di distribuzioni di probabilità e aggiornamenti bayesiani."

Quindi in breve : 
Thompson che usa un metodo probabilistico, ci restituirà lo stesso risultato dell'algoritmo UBC, con la differenza che 
questo lo può restituire con meno passaggi (esempio N=500) rispetto all'altro che è un algoritmo DETERMINISTICO,ovvero
restituisce quello che ha.
QUINDI TRA I 2 MEGLIO USARE QUESTO DI THOMPSON
"""
