
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 6)) 


plt.gca().set_aspect('equal', adjustable='datalim') 

log = np.genfromtxt("Error/Kovasznay/error_k_re=40.0_lin.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'dT = dH')


log = np.genfromtxt("Error/Kovasznay/error_k_re=40.0_quad.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'dT = dH²')






plt.title("Decaimento de erro logarítmico do Fluxo de Kovasznay Estacionário")
plt.legend()
plt.savefig("error_log.png")
#
#
plt.legend(loc='best',framealpha=0.3)

x_annotation = 2.5  # x annotation point on the graph (log scale, so use powers of 10)
y_annotation = -4
slope = 2

plt.annotate(xy=(x_annotation-0.5, y_annotation+0.5),xytext=(x_annotation-0.4, y_annotation-0.35),text="k = 2",)

plt.annotate(xy=(x_annotation-0.5, y_annotation+1),xytext=(x_annotation, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.annotate(xy=(x_annotation-0.5, y_annotation),xytext=(x_annotation, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='-'))
plt.annotate(xy=(x_annotation-0.5, y_annotation+1),xytext=(x_annotation-0.5, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='-'))

plt.plot(x_annotation - 0.5, y_annotation + 1, '.', color='black')  # Start point (head)
plt.plot(x_annotation - 0.5, y_annotation, '.', color='black')  # End point (tail)
plt.plot(x_annotation , y_annotation, '.', color='black')  # End point (tail)
 
plt.grid(True)
#plt.axis('equal')
#plt.xlim(2,6)
#plt.ylim(-8,-2)
plt.xlabel("log(N)")
plt.ylabel("Erro Máximo Absoluto")
plt.show()
plt.close()













plt.figure(figsize=(6, 6)) 


plt.gca().set_aspect('equal', adjustable='datalim') 

log = np.genfromtxt("Error/Kovasznay/errorPressure_k_lin.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'Erro p Δt = Δh')


log = np.genfromtxt("Error/Kovasznay/errorPressure_k.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'Erro p Δt = Δh²')


log = np.genfromtxt("Error/Kovasznay/errorGrad_k_lin.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'Erro ∇p Δt = Δh')


log = np.genfromtxt("Error/Kovasznay/errorGrad_k.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'Erro ∇p Δt = Δh²')




plt.gca().set_aspect('equal', adjustable='datalim') 
plt.grid(True)

plt.title("Decaimento do erro de pressão logarítmica do Fluxo de Kovasznay Estacionário")
plt.legend()
plt.savefig("error_log.png")
#
#
plt.legend(loc='best',framealpha=0.3)

x_annotation = 2.5  # x annotation point on the graph (log scale, so use powers of 10)
y_annotation = -2
slope = 2

plt.annotate(xy=(x_annotation-0.5, y_annotation),xytext=(x_annotation-0.4, y_annotation-0.35),text="k = 2",)

plt.annotate(xy=(x_annotation-0.5, y_annotation+1),xytext=(x_annotation, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.annotate(xy=(x_annotation-0.5, y_annotation),xytext=(x_annotation, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='-'))
plt.annotate(xy=(x_annotation-0.5, y_annotation+1),xytext=(x_annotation-0.5, y_annotation),text="",
             arrowprops=dict(facecolor='black', arrowstyle='-'))

plt.plot(x_annotation - 0.5, y_annotation + 1, '.', color='black')  # Start point (head)
plt.plot(x_annotation - 0.5, y_annotation, '.', color='black')  # End point (tail)
plt.plot(x_annotation , y_annotation, '.', color='black')  # End point (tail)
 

#plt.axis('equal')

plt.xlabel("log(N)")
plt.ylabel("Erro Máximo Absoluto")
plt.show()
plt.close()










