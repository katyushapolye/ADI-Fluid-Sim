import sys
import numpy as np
import matplotlib.pyplot as plt

from os import listdir






N = int(sys.argv[1])
#heatPlot = int(sys.argv[2]) #2d dimensional plot 
#vectorField = int(sys.argv[3]) #if the field is a 2d vector field


#U,V = np.linspace(0,1,N);


    
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)



c = 0
p = 1
for files in listdir('Fields/ScalarFields'):
    filename = files.split('.')[0]
    c = int(filename.split("_")[1])

    
    data = np.genfromtxt('Fields/ScalarFields/' + filename + '.csv', delimiter=',')
    data = data.astype(float)

    fig = plt.figure()
    print("Plotting {:.2f}% ".format((p/ len(listdir('Fields/ScalarFields')))*100),end='\r',)

    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim3d(0,1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('U')
    ax.plot_surface(X, Y, data,cmap='viridis',vmin=0, vmax=1)
    ax.set_title("Iteration {}".format(int(c)))
    plt.savefig('Frames/ScalarFrames/' + filename.split("_")[0] +"_" + str(int(c)) + '.png')
    plt.close(fig) #saves mem





    ax = None
    fig = None    
    data = None
    import gc
    gc.collect()
    p +=1
    #plt.show()



log = np.genfromtxt("Error/erros_conv.csv",delimiter=',')

dH = log[:,1]
error = log[:,3] 
print("Calculating error decay")
order = (np.log(error[-1]) - np.log(error[0])) / (np.log(dH[-1]) - np.log(dH[0]))

plt.plot(-np.log(dH),np.log(error),marker='*',color='black')
plt.title("Error \u0394t = \u0394h- Conv. = {:.4f}".format(order))
plt.show()
