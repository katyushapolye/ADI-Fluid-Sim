import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import Normalize
from tqdm import tqdm
import csv
import pandas as pd
from numba import njit
#What is left...
#Esquema em blocos

#Rodar degrau
    #. Rodar primeiro quadrado
    #. Dividir Nx e Ny

#Caso linear pro SE, comparação.


    



#Definitions
#Grid size
N  = int(sys.argv[1])
#Time Domain, not used for now, steady state problem
t0 = 0
tF = 0
#Space Domain
e0 = 0#-np.pi
eF = 1# np.pi

#Solver info
dH = (eF - e0) / (N-1) #there are 2 dead points because of ease of work in mem alocation, see diagram with me  (raphael)
dT = (dH)
print("dT = {} -- dH = {}".format(dT,dH))
    

#Equations parameters
RE = 1000.0
EPS = 1.0/RE
SIG = (EPS*dT)/(dH**2)



def exact(x,y,t):

    vec_x = -np.cos( x) * np.sin( y) * np.exp((-2 * (1) * t) / RE)
    vec_y = np.sin( x) * np.cos(y) * np.exp((-2 * (1) * t) / RE)

    return np.array([vec_x, vec_y])  # Shape will be (2, N, N)


def exactPressure(x,y,t):
    return (1/4)*(np.cos(2*x)+np.cos(2*y)) * ((np.exp((-4 * (1) * t) / RE)))

def f(x,y,t):
    return 0





def gradP(x,y,t):

    vec_x = 0.5 *  np.sin(2  * x) * np.exp((-4 * (1) * t) / RE)  
    vec_y = 0.5 * np.sin(2  * y) * np.exp((-4 * (1) * t) / RE)  

    return np.array([vec_x, vec_y])  # Shape will be (2, N, N)



def plotVectorField(field, it, name):
    # Offset the cell centers by dH/2
    x = np.linspace(e0 + dH / 2, eF - dH / 2, N - 1)
    y = np.linspace(e0 + dH / 2, eF - dH / 2, N - 1)

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Compute the magnitude of the vector field
    mag = np.sqrt(field[0]**2 + field[1]**2)

    # Plot filled contours for the magnitude of the vector field
    contour = ax.contourf(X, Y, mag, cmap='viridis')

    # Add a colorbar for the contour plot and adjust its location
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Magnitude')

    # Plot vector field with quiver to show the direction
    ax.quiver(X, Y, field[0], field[1], angles='xy', scale_units='xy', scale=5)

    # Ensure the plot is square by setting the aspect ratio
    ax.set_aspect('equal', 'box')

    # Set the title of the plot
    ax.set_title("Iteration {}".format(int(it)))

    # Save the figure
    plt.savefig('Frames/VectorFrames/' + name + "_" + str(int(it)) + '.png')

    # Close the figure to free memory
    plt.close(fig)


def exportVectorField():

    return


def plotPressureField(field, it, name):
    x = np.linspace(e0 - dH / 2, eF + dH / 2, N + 1)  # the line intervals
    y = np.linspace(e0 - dH / 2, eF + dH / 2, N + 1)  # the y intervals

    # Create a meshgrid for the surface plot
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define normalization for consistent color scaling
    normalization = Normalize(vmin=-0.2, vmax=0.2)

    # Plot a 3D surface with constant color scale
    surf = ax.plot_surface(X, Y, field, cmap='viridis', norm=normalization)

    # Set consistent z-axis limits
    ax.set_zlim(-0.2, 0.2)

    # Add a color bar with the same normalization
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Pressure')

    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Pressure')
    ax.set_title(f"Iteration {int(it)}")

    # Save the figure
    plt.savefig(f'Frames/ScalarFrames/{name}_{int(it)}.png')

    # Close the figure to free memory
    plt.close(fig)



#returns to the centered grid layout, receives the u and v stagerred components
def interpolate(u,v):
    U = np.zeros((2,N-1,N-1))
    U[0][:,:] = (u[1:-1,1:] +u[1:-1,:-1])/2.0   #+u[1:-1,2:-1])/2.0 # for each cell center, we interpolate the left and right edge for the u, and top and botton for v, very simple but easy to make mistakes with the indexes
    U[1][:,:] = (v[1:,1:-1] +v[:-1,1:-1])/2.0
    return U

#interpolates the V velocity nodes to the U nodes, we dont interpolate the outer nodes so we just let them in 0, theyre not used anyway
def interpolateVtoU(v):
    vU = np.zeros((N+1,N))
                    #Top 2 nodes            #botton 2
    vU[1:-1,:] = (((v[:-1,:-1] + v[:-1,1:])/2) + ((v[1:,:-1] + v[1:,1:])/2))/2
    return vU
#interpolates the U velocity nodes to the V nodes
def interpolateUtoV(u):
    uV = np.zeros((N,N+1))
    uV[:,1:-1] = (((u[:-1,:-1] + u[:-1,1:])/2) + ((u[1:,:-1] + u[1:,1:])/2))/2
    return uV


#a = Lower Diag, b = Main Diag, c = Upper Diag, d = font\
@njit
def TDMA(a,b,c,d):
    n = len(d)
    w= np.zeros(n-1,float)
    g= np.zeros(n, float)
    p = np.zeros(n,float)
    
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p

@njit
def tridiagonalX(size,vField,i):
    mainDiag = np.zeros(size)
    botton = np.zeros(size-1)
    top = np.zeros(size-1)

    #cctexacttion term
    #ro1 = np.zeros(size)
    ro1 = (vField[0][i,1:-1]*dT)/dH

    #main diag is 1SIG
    mainDiag[:] = 1+SIG


    #botton diag is ro1/2 - sig/2, doest include the last term since its boundary
    botton[:] =  (-ro1[:-1]/4)  -(SIG/2) 

    #top diag is -ro1/2 - sig/2, doesnt include the first term since its boundary
    top[:] =  (ro1[1:] /4)  -(SIG/2) 


    mat = []

    mat.append(botton)
    mat.append(mainDiag)
    mat.append(top)

    return mat


def calculateDivergency(u,v):
    div = np.zeros((N-1,N-1))
    div = ((u[1:-1,1:] - u[1:-1,:-1]) / dH) + ((v[1:,1:-1] - v[:-1,1:-1])/ dH)
    return div

def calculateUPressureGradient(P):
    gPu = np.zeros((N+1,N))
    gPu[:,:] = (P[:,:-1] - P[:,1:]) / dH

    return gPu

def calculateVPressureGradient(P):
    gPv = np.zeros((N,N+1))
    gPv[:,:] = (P[:-1,:] - P[1:,:]) / dH

    return gPv

def plotDivergency(name,divp):
    ITs = np.linspace(0,len(divp),len(divp))
    plt.plot(ITs,divp,c="blue",label="Divergency Value")
    #plt.plot(ITs, np.cumsum(div), c="Red", label="Divergency Cumulative Sum")
    plt.title("Divergency Graph - N = {}".format(N))
    plt.xlabel("Iteration") 
    plt.ylabel("Divergency Value")
    plt.legend()
    plt.savefig('Graphs/Divergency_{}_N={}.png'.format(name,N))
    plt.close()
    return




@njit
def tridiagonalY(size, vField,collum):
    mainDiag = np.zeros(size)
    botton = np.zeros(size-1)
    top = np.zeros(size-1)

    #convection term
    ro2 = np.zeros(size)
    ro2[:] = (vField[1][1:-1, collum]*dT)/dH #only the non-bounds line elements, in Y

    mainDiag[:] = 1+SIG

    botton[:] =  (-ro2[:-1] /4)  -(SIG/2) 

    top[:] =  (ro2[1:] /4)  -(SIG/2) 


    mat = []
    mat.append(botton)   #0
    mat.append(mainDiag) #1
    mat.append(top)      #2

    return mat



#exports the x = 0.5 vector
def exportGhia(filename,Uint,axis):
    with open(filename, 'a') as f:
        axis[0] = 0
        axis[-1] = 1.0
        axisexport  = axis.reshape(1, -1)
        np.savetxt(f, axisexport, delimiter=',', fmt='%f', footer='', comments='')
        #export = np.zeros(N-1)
        export = Uint[0][:,((N)// 2) - 1 ].reshape(1,-1)
        export = np.concatenate(([0.0], export.flatten(), [1.0]))
        export = export.reshape(1,-1)

        np.savetxt(f, export, delimiter=',')
        





# ---------------> SMALL NOTATION COMMENT <---------------
#we use u and v for the vector components, these are in the cells edges
#we use U for the whole vector, this is in the cell center, interpolated back from the edges
#we use x,y for the axis, or X,Y for the whole numpy grid
#our y increases with i




#solving the staggered grid in their points, check notes, this is VERY delicate work, ---> (CHECK THE DIAGRAM, WHICH SHOULD BE IN THE COLAB ATTACHMENT) <----
#note that, in u, there is a extra element in the y axis, both downards and upwards, because of the grid aligment

#in U, our grid is descolated half of dh downards (or upwards, depends in your index definitions) in relation to cell centers
#discrete U grid
xu = np.linspace(e0, eF, N) #the line intervals
yu = np.linspace(e0-(dH/2), (eF+dH/2),N+1)
Xu, Yu = np.meshgrid(xu  , yu) #the complete grid


#exactly the same thing but, for the v, we dislocate in the x axis
xv = np.linspace(e0-(dH/2), (eF+dH/2),N+1) 
yv =  np.linspace(e0, eF, N) 
Xv, Yv = np.meshgrid(xv,yv)

#discrete Pressure grid
xp = np.linspace((e0 - dH/2),(eF + dH/2),N+1)
yp = np.linspace((e0 - dH/2),(eF + dH/2),N+1)
Xp,Yp = np.meshgrid(xp,yp)


#solve the initial condition for u and v, interpolate back to the U 
#            i(y),j(x)
u = np.zeros((N+1,N))
v = np.zeros((N,N+1))

u[-1,:] += 1
#u = exact(Xu,Yu,0)[0]
#v = exact(Xv,Yv,0)[1]



P = np.zeros((N+1,N+1))
#P = exactPressure(Xp,Yp,0)




#=========Solving ADI, assuming i = Y and j = X, Y growing donwards and X growing sideways, normally===========
#Initial conditions
uant = u
vant = v
Pant = P

#variables
umid = np.zeros((N+1,N))
vmid = np.zeros((N,N+1))
Pmid = np.zeros((N+1,N+1))

#pressure grad

gradPu = calculateUPressureGradient(Pant)
gradPv = calculateVPressureGradient(Pant)

#velocity field, for now, it is the exact solution at each point, we need to check how to do it
bu = np.zeros((2,N+1,N))
bv = np.zeros((2,N,N+1))
bu[0] = uant
bu[1] = interpolateVtoU(vant)
bv[0] = vant
bv[1] = interpolateUtoV(uant)

U = interpolate(u,v)
plotVectorField(U,0,"Interpolated")

plotPressureField(P,0,"Pressure")
plotPressureField(Pant,0,"Pressure_Exact")


#control variables
IT = 1

ITFINAL = ((tF-dT)/dT)
time = 0
time = time+dT/2 

#error variabes

gradError = []
error = []
pError = []
div = []
bforeDiv = []








#BAR = tqdm(total=ITFINAL, desc="Progress")

stable = False
while stable == False:

    #BAR.update(1)


    #bounds conditions in all points
    umid[0,:] =   0 #exact(xu,(e0 - dH/2) ,time)[0] #upper bounds
    umid[-1,:] =  1#exact(xu, (eF + dH/2),time)[0] #lower bounds
    umid[:,0] =   0#exact(e0,yu,time)[0]          #left side
    umid[:,-1] =  0#exact(eF,yu,time)[0]           #right side
#
    vmid[0,:] =   0#exact(xv,(e0) ,time)[1] #upper bounds
    vmid[-1,:] =  0#exact(xv,(eF) ,time)[1] #lower bonds
    vmid[:,0] =   0#exact(e0-(dH/2),yv ,time)[1] 
    vmid[:,-1] =  0#exact(eF+(dH/2),yv ,time)[1]
#
    u[0,:] =      0#exact(xu,(e0 - (dH/2)) ,time+ dT/2)[0] #upper bounds
    u[-1,:] =     1#exact(xu, (eF + (dH/2)),time+ dT/2)[0] #lower bounds
    u[:,0] =      0#exact(e0,yu,time+ dT/2)[0]          #left side
    u[:,-1] =     0#exact(eF,yu,time+ dT/2)[0]           #right side
#
    v[0,:] =      0 #exact(xv,(e0) ,time+ dT/2)[1] #upper bounds
    v[-1,:] =     0 #exact(xv,(eF) ,time+ dT/2)[1] #lower bonds
    v[:,0] =      0#exact(e0-(dH/2),yv ,time + dT/2)[1] 
    v[:,-1] =     0#exact(eF+(dH/2),yv ,time+ dT/2)[1]

    P = np.zeros((N+1,N+1))
    P[0,:] =     0#exactPressure(xp,(e0-dH/2),time)
    P[-1,:] =    0#exactPressure(xp,(eF+dH/2),time)
    P[:,0] =     0#exactPressure((e0-dH/2),yp,time)
    P[:,-1]=     0#exactPressure((eF+dH/2),yp,time)


    #velocity field, for now, it is the exact solution at each point, we need to check how to do it
    bu = np.zeros((2,N+1,N))
    bv = np.zeros((2,N,N+1))
    bu[0] = uant#exact(Xu,Yu,time)[0]
    bu[1] = interpolateVtoU(vant)#exact(Xv,Yv,time)[1])


    bv[1] = vant#exact(Xv,Yv,time)[1]
    bv[0] = interpolateUtoV(uant)#exact(Xu,Yu,time)[0])


    #implicit in the x direction
    #for u, we need to go from one to N
    for i in range(1,N):
        ro2 = np.zeros(N) #velocity term
        ro1 = np.zeros(N)

        ro2[:] = (bu[1][i,:] * dT) /dH
        ro1[:] = (bu[0][i,:] * dT) /dH

        #font, we have the same number in x
        font = np.zeros(N-2) 
        font[:] = (1-SIG)*uant[i,1:-1] + ((SIG/2.0 - ro2[1:-1]/4) * uant[i+1,1:-1])+ ((SIG/2.0 + ro2[1:-1]/4) * uant[i-1,1:-1]) 
        font[0] +=  umid[i,0] *  (ro1[1]/4 + SIG/2)#border
        font[-1] += umid[i,-1] *  (-ro1[-2]/4 + SIG/2) #border

        matrix = tridiagonalX(N-2,bu,i)
        umid[i,1:-1] = TDMA(matrix[0],matrix[1],matrix[2],font)
       
    #for v, we have the usual limits
    for i in range(1,N-1):
        ro2 = np.zeros(N+1) #velocity term
        ro1 = np.zeros(N+1)

        ro2[:] = (bv[1][i,:] * dT) /dH
        ro1[:] = (bv[0][i,:] * dT) /dH

        #our font here is bigger because we have one more term
        font = np.zeros(N-1)
        font[:] = (1-SIG)*vant[i,1:-1] + ((SIG/2.0 - ro2[1:-1]/4) * vant[i+1,1:-1])+ ((SIG/2.0 + ro2[1:-1]/4) * vant[i-1,1:-1]) 
        font[0] +=  vmid[i,0] * (ro1[1]/4 + SIG/2)#border
        font[-1] += vmid[i,-1]*(-ro1[-2]/4 + SIG/2) #border

        matrix = tridiagonalX(N-1,bv,i)
        vmid[i,1:-1] = TDMA(matrix[0],matrix[1],matrix[2],font)





    #finished first half step, so we can now increase our time, and repeat the same process
    time = time+dT/2
    #velocity field, for now, it is the exact solution at each point, we need to check how to do it
    bu = np.zeros((2,N+1,N))
    bv = np.zeros((2,N,N+1))
    bu[0] = umid#exact(Xu,Yu,time)[0]
    bu[1] = interpolateVtoU(vmid)#exact(Xv,Yv,time)[1])


    bv[1] = vmid#exact(Xv,Yv,time)[1]
    bv[0] = interpolateUtoV(umid)#exact(Xu,Yu,time)[0])

    #ppgmc@ice.ufjf.br

    #for the u component, in the y axis, we have the usual size
    for j in range(1,N-1):
        ro2 = np.zeros(N+1) #velocity term
        ro1 = np.zeros(N+1)
        ro1[:] = (bu[0][:, j] * dT)/dH
        ro2[:] = (bu[1][:, j] * dT)/dH

        #we have now a extra term in the font
        font = np.zeros(N-1)
        font[:] = (1-SIG)* umid[1:-1,j] + ((SIG/2.0 - ro1[1:-1]/4)* umid[1:-1,j+1]) + ((SIG/2.0 + ro1[1:-1]/4)* umid[1:-1,j-1]) 
        font[0] += u[0,j]* (ro2[1]/4 + SIG/2)
        font[-1] += u[-1,j]*(-ro2[-2]/4 + SIG/2) 

        matrix = tridiagonalY(N-1,bu,j)
        u[1:-1,j] = TDMA(matrix[0],matrix[1],matrix[2],font)

    for j in range(1,N):
        ro2 = np.zeros(N) 
        ro1 = np.zeros(N)
        ro1[:] = (bv[0][:, j] * dT)/dH
        ro2[:] = (bv[1][:, j] * dT)/dH


        font = np.zeros(N-2)
        font[:] = (1-SIG)* vmid[1:-1,j] + ((SIG/2.0 - ro1[1:-1]/4)* vmid[1:-1,j+1]) + ((SIG/2.0 + ro1[1:-1]/4)* vmid[1:-1,j-1]) 
        font[0] += v[0,j]* (ro2[1]/4 + SIG/2)
        font[-1] += v[-1,j]*(-ro2[-2]/4 + SIG/2) 

        matrix = tridiagonalY(N-2,bv,j)
        v[1:-1,j] = TDMA(matrix[0],matrix[1],matrix[2],font)

    #the algorithm has finished the step, so wwe increment the time and swap everything
    #also, we plot the solution, but we have to interpolate it back first
    time+= dT/2

    #solving the pressure now that we have a non-conservative velocity field
    #the procesure is based in the helmholtz decomposition and 
    #the relation between the pressure and velocity that arises when
    #we take the divergence of both sides in the
    #momentum equations - > we have -> laplacian P =  div u
    #whe should be able to solve for P (maybe we can even solve the for the gradient directly)

    #solving P,using the linalg solver for now, we have a extra cell at every edge, because we need the grad, ask questions
    #solve p
    ##pressure solving, final half step
    ###border conditions, tecnically, it should work, so...
    
    B = calculateDivergency(u,v)


    P = np.zeros((N+1,N+1))
    itCount = 0
    
    while(True):
        Pant[:,0] =     Pant[:,1]   #+ (dH/dT)*( u[:,0]-u[:,1] )
        Pant[:,-1] =    Pant[:,-2]  #+ (dH/dT)*(  u[:,-1]-u[:,-2])
        Pant[0,:] =     Pant[1,:]   #+ (dH/dT)*( v[0,:]-v[1,:])
        Pant[-1,:] =    Pant[-2,:]  # + (dH/dT)*(v[-2,:] - v[-1,:])

        itCount +=1
        for i in range(1, N):
            for j in range(1, N):
                P[i, j] =  (Pant[i-1, j]+Pant[i+1, j]+Pant[i, j-1] + Pant[i, j+1] + (B[i-1, j-1]*((dH**2)/(dT))))/(4)
        if(np.max(np.abs(P - Pant)) < 1e-10 and itCount > 10): #se a dif entre ambos for menor que -12, parar, tambem so aumentar pra 9 pra ser mais estavel
            break
        Pant = P





    ##aprox gradP
    #P = exactPressure(Xp,Yp,time)
    gradPu = calculateUPressureGradient(P)
    gradPv = calculateVPressureGradient(P)
    #subtract gradP from solution 
    u[1:-1,1:-1] = u[1:-1,1:-1] - (dT)*gradPu[1:-1,1:-1]
    v[1:-1,1:-1] = v[1:-1,1:-1] - (dT)*gradPv[1:-1,1:-1]

 



    


    Uant = interpolate(uant,vant)
    U = interpolate(u,v)

    diff = np.abs(U - Uant)
    print("Pressure converged in {} iterations --- Dif {} -- It = {} -- t = {}".format(itCount,np.max(diff),IT,time),end="\r")
    if(np.max(diff) < 1e-6 and IT > 10): 
        print("SIMULATION STABLE -- ENDING")
        stable = True

    




    uant = u
    vant = v
    umid = np.zeros((N+1,N))
    vmid = np.zeros((N,N+1))
    u = np.zeros((N+1,N))
    v = np.zeros((N,N+1))

    Pant = P
    P = np.zeros((N+1,N+1))



    #error calculation, we calculate it before the interpolation, in each component
    
    bforeDiv.append(np.sum(np.abs(B)))
    div.append(np.sum(np.abs(calculateDivergency(uant,vant)))) #talvez dividir por N
    #gradError.append(np.max(gradPu -gradP(Xu,Yu,time)[0]))
    #gradError.append(np.max(gradPv -gradP(Xv,Yv,time)[1]))

    #uexact = exact(Xu,Yu,time-dT/2)[0] #we are half a dt foward, so we just subtract it 
    #vexact = exact(Xv,Yv,time-dT/2)[1] #we are half a dt foward, so we just subtract it 
    #error.append(np.max(np.abs(uexact - uant)))
    #error.append(np.max(np.abs(vexact - vant)))
    #pError.append(np.max(Pant - exactPressure(Xp,Yp,time)))







    #Plotting, uncomment each plot separately if you wanna actually run this, this is very heavy
    #plot the aproximate and the exact solution just for comparison, interpolated to the same point
    #U = interpolate(uant,vant)
    #plotVectorField(U,IT,"Interpolated")
    #U = interpolate(uexact,vexact)
    #plotVectorField(U,IT,"InterpolatedExact")
    ###and also plot the individual points
    ##plotScalarFieldX(uant,IT,"ScalarU")
    ##plotScalarFieldX(uexact,IT,"ScalarUExact")
    ###and also the pressure gradient interpolated
    ##PressureInterpolated = interpolate(gradPu,gradPv)
    ##plotVectorField(PressureInterpolated,IT,"Pressure_Gradient")
    ##PressureInterpolated = interpolate(gradP(Xu,Yu,time)[0],gradP(Xv,Yv,time)[1])
    #plotVectorField(PressureInterpolated,IT,"Pressure_Gradient_Exact")
    #plotPressureField(exactPressure(Xp,Yp,time),IT,"Pressure_Exact")
    #plotPressureField(Pant,IT,"Pressure")

    

    IT +=1


##after the main loop is finished, we just take the maximum of the both components max error in each interaction
#with open("Error/error.csv", 'a') as file:
#    file.write(f'{N},{max(error)}\n')
#
#with open("Error/errorGrad.csv", 'a') as file:
#    file.write(f'{N},{max(gradError)}\n')
#
#with open("Error/errorPressure.csv", 'a') as file:
#    file.write(f'{N},{max(pError)}\n')
#plot the divergency graph
#plotDivergency("Lid Driven_After",div)
#plotDivergency("Lid_Driven_Before",bforeDiv)


U = interpolate(uant,vant)
plotVectorField(U,IT,"Interpolated_Final")
df = pd.DataFrame(interpolate(uant,vant)[0],columns=xp[1:-1],index=yp[1:-1]) #interpolated export of the final field
df.to_csv("Field_Export_U.csv")



exportGhia("Guia_r={}_n={}.txt".format(RE,N),U,yp)





