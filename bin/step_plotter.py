import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import numpy as np
from scipy.interpolate import make_interp_spline





plt.ylim((0, 16.5))
plt.xlim((0, 11.0))
plt.xlabel('Rₑ x 10²')
plt.ylabel('xL')
plt.title('Zona de Recirculação para Reynolds Até 1000')

# Set major and minor ticks
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.gca().set_aspect(0.5) 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

Re = np.array([1, 
               2,
               3.98,
               6.0,
               10.0])
XlSim = np.array([3.0,
                  5.0,
                  7.0,
                  10.0,
                  13.0])

uncertainty = [0.5, 0.5,0.5,0.5,0.5]
plt.errorbar(Re, XlSim, yerr=uncertainty, fmt='s',
             ecolor='gray', elinewidth=1, capsize=3,label='Dados Simulados')

Re_smooth = np.linspace(Re.min(), Re.max(), 200)

XlSim_smooth = make_interp_spline(Re, XlSim,k=1)(Re_smooth)
plt.plot(Re_smooth, XlSim_smooth,color='#1f77b4',linestyle='--')





Re = [1, 1.5, 3.1, 4.0, 4.5, 6.2,6.5,7.0,8.0]
XlExp = [3.0, 4.0, 6.4, 8.3, 8.6, 11.2,12.1,12.8,14.1]

plt.scatter(Re, XlExp, marker='o', facecolors='none',edgecolors='orange',label='Dados Experimentais',s=100.0)


coefficients = np.polyfit(Re, XlExp, deg=2)  
regression_line = np.poly1d(coefficients)
Re_range = np.linspace(min(Re), max(Re), 100)
Xl_fit = regression_line(Re_range)
plt.plot(Re_range, Xl_fit, color='Orange', linestyle='--',linewidth=0.5)


plt.legend()
plt.show()
