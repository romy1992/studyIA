import math

import matplotlib.pyplot as plt
import pandas as pd

# Leggo
dataset = pd.read_csv('../Ads_CTR_Optimisation.csv')

# Creo le variabili
N = 10000  # Numeri di righe(Utenti)
d = 10  # Numeri di colonne (Annunci)
ads_selected = []  # Annunci selezionati
numbers_of_selections = [0] * d  # Ni(n) : Num annunci selezionati -> Crea una lista di 0 tanti quanto è la variabile d
sums_of_rewards = [0] * d  # Ri(n) : Somma ricompensa -> Crea una lista di 0 tanti quanto è la variabile d
total_reward = 0  # Ricompensa totale
for n in range(0, N):  # Per ogni riga
    ad = 0  # Annuncio selezionato
    max_upper_bound = 0  # Valore massimo di CONFIDENZA
    for i in range(0, d):  # Per ogni colonna
        if numbers_of_selections[i] > 0:  # Se l'annuncio è stato selezionato
            # Calcolo la media dei selezionati: Ri(n)/Ni(n)
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            """
            Applico la formula per recuperare il valore delta: √(3/2 * log(n) / Ni(n))
            IL MATH.LOG(N) NEL PRIMO GIRO SARA' 0 E QUINDI DARA' VALORI INFINITI.PER QUESTO SI AGGIUNGE IL +1 
            """
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i  # Calcolo la somma finale
        else:  # Altrimenti imposta un valore di default
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i  # Questo è il valore di CONFIDENZA maggiore della riga e colonna e quindi selezionabile (Annuncio)

    ads_selected.append(ad)  # Aggiungo alla lista di annunci i migliori di ogni riga
    numbers_of_selections[ad] += 1  # Incremento di 1 l'elemento selezionato della colonna selezionata(Annuncio)
    reward = dataset.values[n, ad]  # Ricompensa
    # Aggiungo alla lista delle ricompense la ricompensa appena ottenuta di una determinata colonna(Annuncio) e la sommo
    sums_of_rewards[ad] += reward
    total_reward += reward  # Sommo il totale delle ricompense

print(ads_selected)
print(numbers_of_selections)
print(sums_of_rewards)
print(total_reward)

# Istogramma con pyplot
plt.hist(ads_selected)
plt.title('Istogramma degli annunci selezionati')
plt.xlabel('Annunci')
plt.ylabel("Numeri dell'annuncio selezionato")
plt.show()

"""
UBC fà parte di quelli algoritmi di "reinforcement learning" , utilizzati come base , per esempio, su come far camminare
un robot.E' un algoritmo DETERMINISTICO che sceglie un'azione casuale per massimizzare la ricompensa attesa.
In questo esempio mostra 10 colonne (che sono gli annunci) e 10000 righe (che sono le scelte degli utenti dove 0 indica 
che quell'annuncio non è stato selezionato da quell'utente e 1 invece è stato selezionato).
Con questa logica lui cercherà mi mostrare sempre l'annuncio più desiderato che man mano verrà addestrato.
"Deterministico, basato su calcoli matematici di medie e bound di confidenza."
"""
