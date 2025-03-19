import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to compute the average from CSV




log = np.genfromtxt("parallel.csv",delimiter=',')
N_values = log[0:,0]
parallel_averages = log[0:,1]

log = np.genfromtxt("sequencial.csv",delimiter=',')
N_values = log[0:,0]
serial_averages = log[0:,1]


# Plotting
plt.figure(figsize=(10, 6))

# Plot parallel and serial averages as curves
plt.plot(N_values**2, parallel_averages, marker='o', label='Paralelo', linestyle='-')
plt.plot(N_values**2, serial_averages, marker='o', label='Sequencial', linestyle='-')

# Add annotations with percentage difference and vertical lines



# Labels and title
plt.xlabel('N²')
plt.ylabel('Segundos')
plt.title('Tempo Médio de Resolução da Iteração no Degrau [0,1] x [0,15]')
plt.legend()
plt.grid(True)
plt.savefig("time")

plt.show()
