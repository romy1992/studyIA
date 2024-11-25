import math
import matplotlib.pyplot as plt
import pandas as pd

# Leggo
dataframe = pd.read_csv('Promotions_Offers_Simulation.csv')

# Creo la struttura di reinforcement
N = 500
prom = 5
number_selected = []
numbers_of_selected = [0] * prom
sums_of_reward = [0] * prom
total_reward = 0
for n in range(0, N):
    p = 0
    max_num_confidence = 0
    for i in range(0, prom):
        if numbers_of_selected[i] > 0:
            # Applico la formula
            average_reward = sums_of_reward[i] / numbers_of_selected[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / numbers_of_selected[i])
            num_confidence = average_reward + delta_i
        else:
            num_confidence = 1e400

        if num_confidence > max_num_confidence:
            max_num_confidence = num_confidence
            p = i
    number_selected.append(p)
    numbers_of_selected[p] += 1
    reward = dataframe.values[n, p]
    sums_of_reward[p] += reward
    total_reward += reward


plt.hist(number_selected)
plt.show()
