import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to compute the average from CSV
def compute_average(filename):
    df = pd.read_csv(filename, header=None)
    return df[1].mean()

# Define N values and corresponding labels
N_values = [16,32,64,128]
parallel_averages = []
serial_averages = []

for N in N_values:
    parallel_avg = compute_average(f'Error/time_parallel_n={N}.csv')
    serial_avg = compute_average(f'Error/time_serial_n={N}.csv')

    parallel_averages.append(parallel_avg)
    serial_averages.append(serial_avg)

# Plotting
plt.figure(figsize=(10, 6))

# Plot parallel and serial averages as curves
plt.plot(N_values, parallel_averages, marker='o', label='Paralelo', linestyle='-')
plt.plot(N_values, serial_averages, marker='o', label='Sequencial', linestyle='-')

# Add annotations with percentage difference and vertical lines
for i, N in enumerate(N_values):
    parallel_avg = parallel_averages[i]
    serial_avg = serial_averages[i]


    
    # Calculate percentage difference relative to the highest value
    max_avg = max(parallel_avg, serial_avg)
    percentage_diff = (((serial_avg ) / parallel_avg) -1.0) * 100
    
    # Annotate near the midpoint between the points
    mid_x = N
    mid_y = (parallel_avg + serial_avg) / 2

    # Draw vertical line connecting the parallel and serial points
    plt.vlines(x=mid_x, ymin=min(parallel_avg, serial_avg), ymax=max(parallel_avg, serial_avg), color='gray', linestyle='--')

    plt.annotate(f'{percentage_diff:.1f}%', 
                 xy=(mid_x, mid_y), 
                 xytext=(0, 10), 
                 textcoords='offset points', 
                 ha='center', 
                 fontsize=10, 
                 color='black')

# Labels and title
plt.xlabel('N')
plt.ylabel('Segundos')
plt.title('Tempo Médio de Resolução da Iteração no Degrau [0,1] x [0,15]')
plt.legend()
plt.grid(True)

plt.show()
