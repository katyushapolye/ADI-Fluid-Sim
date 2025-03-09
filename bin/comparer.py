
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
aprox = []  
axis = []

def readVectors(filename):
    # Load the data from the file
    data = np.loadtxt(filename, delimiter=',')
    
    # Assuming the file has two rows: the first row for 'axis' and the second for 'Uint'
    axis = data[0]  # First row
    Uint = data[1]  # Second row

    return axis, Uint




xghia =  [1.00000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
              0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 
              0.0000]

#guia re=100
ghia = [
    1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 
    0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, 
    -0.06434, -0.04775, -0.04192, -0.03717, 0.00000
]

axis,aprox = readVectors("Graphs/Ghia_re100/Guia_r=100.0_n=16.txt")
plt.plot(axis, aprox, label='N = 16', linewidth=0.75)

axis,aprox = readVectors("Graphs/Ghia_re100/Guia_r=100.0_n=32.txt")
plt.plot(axis, aprox, label='N = 32', linewidth=0.75)

axis,aprox = readVectors("Graphs/Ghia_re100/Guia_r=100.0_n=64.txt")
plt.plot(axis, aprox, label='N = 64', linewidth=0.75)

axis,aprox = readVectors("Graphs/Ghia_re100/Guia_r=100.0_n=128.txt")
plt.plot(axis, aprox, label='N = 128', linewidth=0.75)


#guia re=1000
#ghia = [
#1.00000,0.65928,0.57492,0.51117,0.46604,0.33304,
#0.18719,0.05702,-0.06080,-0.10648,
#-0.27805,-0.38289,-0.29730,-0.22220,-0.20196,-0.18109,
#0.00000]
#
#axis,aprox = readVectors("Graphs/Ghia_re1000/Guia_r=1000.0_n=64.txt")
#plt.plot(axis, aprox, label='N = 64', linewidth=0.75)
#
#axis,aprox = readVectors("Graphs/Ghia_re1000/Guia_r=1000.0_n=128.txt")
#plt.plot(axis, aprox, label='N = 128', linewidth=0.75)
#
#axis,aprox = readVectors("Graphs/Ghia_re1000/Guia_r=1000.0_n=256.txt")
#plt.plot(axis, aprox, label='N = 256', linewidth=0.75)





plt.scatter(xghia, ghia, facecolor='none', edgecolor='red', label='Ghia et al.', marker='.')
plt.title('Aproximação para Re = 1000.0 - dT = dH'.format(len(axis)-1))
plt.xlabel('y')
plt.ylabel('u')

plt.legend()  # Add a legend


plt.ylim(-0.4, 1)  # Set the y-axis limits
plt.xlim(0, 1)     # Set the x-axis limits
plt.savefig("Graphs/Ghia_re=100.png".format(len(axis)-1), dpi=300, bbox_inches='tight')
plt.show()
