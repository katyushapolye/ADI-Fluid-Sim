import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))  # Adjusted figure size for better visualization
plt.gca().set_aspect('auto', adjustable='box')

# Load data for ADI Average Time
file = np.genfromtxt("Error/time.csv", delimiter=',')
N_adi = file[0:, 0]
time_adi = file[0:, 1]

# Load data for ADI + SE Average Time
file = np.genfromtxt("Error/time_explicit.csv", delimiter=',')
N_se = file[0:, 0]
time_se = file[0:, 1]

# Plot the curves
plt.plot(N_adi**2, time_adi, marker='.', label='ADI Average Time')
plt.plot(N_se**2, time_se, marker='.', label='ADI + SE Average Time')

# Calculate the differences between the two curves
differences = np.abs(time_adi - time_se)

# Plot the difference bars
plt.bar(N_adi**2, differences, width=5000, alpha=0.3, color='gray', label='Difference')

# Add title, labels, and legend
plt.title("Average Iteration CPU Time - Rₑ = 1000")
plt.xlabel("Number of grid cells")
plt.ylabel("Time (ms)")
plt.xlim(0, 2e6)
plt.ylim(0, 10)
plt.grid()
plt.legend()

plt.show()