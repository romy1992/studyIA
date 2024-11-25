import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Recupero il csv
df = pd.read_csv('Salary_Data.csv')

# Dividiamo X e y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Pulire gli eventuali valori null o codificare le stringhe
# Nulla per questo esempio

# Split dei vari parametri tra train e test per darli in pasto al modello di addestramento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Prima di addestrare(e quindi DOPO aver fatto lo split)
# bisogna capire se applicare il Future Scaling (StandardScaler consigliato) solo sui parametri di entrata(X)
# Applicando il fit_transform a X_train e SOLO transform a X_test

# Linea di regressione per prevedere valori come il salario
regressor = LinearRegression()
# Si addestra il modello con i parametri di TRAIN
regressor.fit(X_train, y_train)

# Prevedere i risultati per X_test(QUESTO è IL PARAMETRO GIUSTO PER LA PREVISIONE)
y_predict = regressor.predict(X_test)

# Grafico di attendibilità per i dati di Training

# Questo metodo creerà i punti rossi che sono i valori REALI del salario che sono i valori di TRAIN
plt.scatter(X_train, y_train, color='red')
# Retta di regressione che si avvicina ai numeri reali
plt.plot(X_train, regressor.predict(X_train), color='blue')
# Titolo
plt.title('Salario VS Esperienza')
# Inserire la label da visualizzare sull'asse X
plt.xlabel('Anni di esperienza')
# Inserire la label da visualizzare sull'asse y
plt.ylabel('Salario')
# Mostra la funzione
plt.show()

# Grafico di attendibilità per i dati di Test

# Questo metodo creerà i punti rossi che sono i valori REALI del salario che sono i valori di TRAIN
plt.scatter(X_test, y_test, color='red')
# Retta di regressione che si avvicina ai numeri reali (Uguale alla retta della grafico di Training)
plt.plot(X_train, regressor.predict(X_train), color='blue')
# Titolo
plt.title('Salario VS Esperienza')
# Inserire la label da visualizzare sull'asse X
plt.xlabel('Anni di esperienza')
# Inserire la label da visualizzare sull'asse y
plt.ylabel('Salario')
# Mostra la funzione
plt.show()

# Fare una singola previsione (ad esempio lo stipendio di un dipendente con 12 anni di esperienza)
print(regressor.predict([[12]]))

# Ottenere l'equazione di regressione lineare finale con i valori dei coefficienti
print(regressor.coef_)
print(regressor.intercept_)

"""
La LinearRegression viene utilizzata per predire valori continui (infiniti) come :
Altezza o peso di una persona (può essere qualsiasi numero, ad esempio 68,2 kg, 68,25 kg, ecc.)
Temperatura (ad esempio, 23,5°C, 23,55°C)
Prezzo di un prodotto (può essere 19,99€, 19,999€, ecc.)
In pratica, quando parliamo di un valore continuo, stiamo considerando dati che non sono limitati a categorie specifiche
o a numeri interi, ma possono avere qualsiasi valore numerico in un intervallo definito. 
Questo si differenzia dai valori discreti, che invece possono assumere solo specifici valori,
come numeri interi (es. conteggio di persone o oggetti).

Quindi alla domanda :

Dovremmo usare la regressione lineare semplice per prevedere il vincitore di una partita di calcio?
E' Falso perché qui si intende un numero tra 1 -1 o 2 e quindi non continuo
 
Regressione lineare semplice: Una sola variabile indipendente (x) 
viene utilizzata per prevedere la variabile dipendente (y). 
L'equazione è di solito: y=β0+β1x+ϵ

dove:
y è la variabile dipendente.
x è la variabile indipendente.
β0 è l'intercetta.(è come una sorta di "valore di base" da cui il modello parte per poi aggiungere (o sottrarre) 
    gli effetti delle variabili indipendenti sul valore previsto di 𝑦)
β1 è il coefficiente di regressione.
ϵ è il termine di errore.
 """
