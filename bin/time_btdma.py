
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(6, 6)) 
plt.gca().set_aspect('auto', adjustable='box')

file = np.genfromtxt("Error/ExptimeBTDMA.csv",delimiter=',')
N = file[0:,0]
time = file[0:,1]

plt.plot(N,time,marker='.',label = 'BTDMA')


file = np.genfromtxt("Error/ExptimeTDMA.csv",delimiter=',')
N = file[0:,0]
time = file[0:,1]

plt.plot(N,time,marker='.',label = 'TDMA')

file = np.genfromtxt("Error/ExptimeBTDMA_SYM.csv",delimiter=',')
N = file[0:,0]
time = file[0:,1]
plt.plot(N,time,marker='.',label = 'BTDMA - SYM')

plt.title("Solve Time for Linear System of Size NxN")

plt.legend()
plt.xlabel("N")
plt.ylabel("time (ns)")
#plt.xlim(0,2e6)
#plt.ylim(0,10)
plt.grid()

plt.show()