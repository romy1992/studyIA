import pandas as pd
from apyori import apriori

# Leggo
dataset = pd.read_csv('../Market_Basket_Optimisation.csv', header=None)

# Aggiungo in un array tutte le colonne e righe
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Inizio con l'addestramento di apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_lenght=2,
                max_length=2)
"""
transactions = transactions:
transactions è il dataset di input che contiene tutte le transazioni. 
Ogni transazione è rappresentata come una lista di item acquistati insieme.
L'algoritmo analizza queste transazioni per trovare pattern frequenti.
Esempio di una transazione: ['pane', 'burro', 'latte'].

min_support = 0.003:
Questo parametro specifica la soglia minima di supporto richiesta per considerare un itemset come frequente.
Supporto indica quanto spesso un particolare itemset appare nel dataset rispetto al totale delle transazioni.
In questo caso, 0.003 significa che l'itemset deve comparire almeno nello 0.3% delle transazioni totali per essere considerato.

min_confidence = 0.2:
Questo parametro indica la soglia minima di confidenza per una regola di associazione.
La confidenza misura la probabilità che l'item 𝐵 sia acquistato quando l'item A è già stato acquistato.
In questo caso, 0.2 significa che la confidenza della regola deve essere almeno del 20%.

min_lift = 3:
Specifica la soglia minima di lift per una regola di associazione.
Il lift misura quanto è forte l'associazione tra gli item rispetto alla loro probabilità di verificarsi indipendentemente.
In questo caso, 3 significa che il lift della regola deve essere almeno 3, il che indica una correlazione positiva significativa tra gli item.

min_length = 2:
Definisce la lunghezza minima degli itemset che l'algoritmo deve considerare.
In questo caso, 2 significa che gli itemset devono contenere almeno 2 item per essere considerati nelle regole di associazione.

max_length = 2:
Definisce la lunghezza massima degli itemset che l'algoritmo deve considerare.
In questo caso, 2 significa che gli itemset devono contenere al massimo 2 item. 
In pratica, si cercano associazioni tra coppie di item.
"""
results = list(rules)
print(results)


# Il metodo sotto è pratico perché sarà sempre identico a questo se bisogna usarlo
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


results_in_dataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

print(results_in_dataFrame)

# Se si vuole riordinarlo per colonna specifica
print(results_in_dataFrame.nlargest(columns='Support', n=10))
"""
E' la versione semplificata di Apriori :
Quando usare Eclat o Apriori:
Eclat è più efficace quando si lavora con dataset densi 
(dove molti item sono presenti insieme nelle transazioni) e si ha bisogno di una ricerca più veloce degli itemset frequenti.
Apriori è preferibile quando si desidera un algoritmo più comprensibile e si sta 
lavorando con dataset più sparsi o con una rappresentazione orizzontale.

In sintesi:
Eclat è un algoritmo più efficiente per dataset grandi e densi, grazie al suo approccio verticale.
Apriori è più intuitivo e semplice da comprendere, ma può essere computazionalmente più pesante per dataset complessi.

Risponde alle seguenti domande :
Apriori può essere visto come "le persone che hanno acquistato hanno anche acquistato"
Nel nostro scenario/esempio - Se a qualcuno è piaciuto il film 1, gli è piaciuto anche il film 2
"""
