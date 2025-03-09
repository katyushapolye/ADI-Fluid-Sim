
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 6)) 


plt.gca().set_aspect('equal', adjustable='datalim') 

log = np.genfromtxt("Error/Quadratic/error_quadratic_re100.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'dT=dH² Rₑ = 100')

log = np.genfromtxt("Error/Quadratic/error_quadratic_re1000.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'dT=dH² Rₑ = 1000')


log = np.genfromtxt("Error/Quadratic/error_quadratic_re10000.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'dT=dH² Rₑ = 10000')
plt.grid(True)

plt.gca().set_aspect('equal', adjustable='datalim') 

log = np.genfromtxt("Error/Linear/error_linear_re100.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'dT=dH Rₑ = 100')

log = np.genfromtxt("Error/Linear/error_linear_re1000.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'dT=dH Rₑ = 1000')


log = np.genfromtxt("Error/Linear/error_linear_re10000.csv",delimiter=',')
dH = log[0:,0]
error = log[0:,1]
plt.plot(np.log(dH),np.log(error),marker='.',label = 'dT=dH Rₑ = 10000')
plt.grid(True)






plt.title("Decaimento de erro logarítmico no vórtice de Taylor-Green")
plt.legend()
plt.savefig("error_log.png")
#
#
plt.legend(loc='best',framealpha=0.3)

x_annotation = 2.5  # x annotation point on the graph (log scale, so use powers of 10)
y_annotation = -7
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
 

#plt.axis('equal')
plt.xlim(2,6)
plt.ylim(-8,-2)
plt.xlabel("log(N)")
plt.ylabel("Erro Máximo Absoluto")
plt.show()
plt.close()
#
#
#
#log = np.genfromtxt("Error/errorGrad.csv",delimiter=',')
#dH = log[0:,0]
#error = log[0:,1]
#print("Calculating error decay")
#order = -(np.log(error[-1]) - np.log(error[0])) / (np.log(dH[-1]) - np.log(dH[0]))
#plt.plot(np.log(dH),np.log(error),marker='*',color='blue',label='Logarithmic Error in \u2207P')
#
#
#log = np.genfromtxt("Error/errorPressure.csv",delimiter=',')
#dH = log[0:,0]
#error = log[0:,1]
#print("Calculating error decay")
#order = -(np.log(error[-1]) - np.log(error[0])) / (np.log(dH[-1]) - np.log(dH[0]))
#plt.plot(np.log(dH),np.log(error),marker='*',color='red',label='Logarithmic Error in P')
#plt.title("Error  P  and \u2207P - \u0394t = \u0394h/2π")
#plt.xlabel('N')
#plt.ylabel('Error')
#plt.legend()
#plt.savefig("error_log_P.png")
#plt.close()





