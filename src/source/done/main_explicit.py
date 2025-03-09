import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import Normalize
from tqdm import tqdm


#What is left...
#aprox the convective/reactive term explicitly
#throw it in the font
#solve for the difusion equation, remove from the matrix the convective terms
from time import process_time


#Definitions
#Grid size
N  = int(sys.argv[1])


#Time Domain
t0 = 0
tF = 1
#Space Domain
e0 = -np.pi
eF =  np.pi

#Solver info
dH = (eF - e0) / (N-1)  
dT = (dH)/(2*np.pi)
    

#Equations parameters
RE = 10000.0
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



def exactDivergency(x,y,t):
    return 0

def exactGradP(x,y,t):

    vec_x = 0.5 *  np.sin(2  * x) * np.exp((-4 * (1) * t) / RE)  
    vec_y = 0.5 * np.sin(2  * y) * np.exp((-4 * (1) * t) / RE)  

    return np.array([vec_x, vec_y])  # Shape will be (2, N, N)



def plotVectorField(field,it,name):
    #we need to offset a bit the cell centers by dh/2
    x = np.linspace(e0+dH/2, eF-dH/2, N-1)
    y = np.linspace(e0+dH/2 ,eF-dH/2, N-1)

    X, Y = np.meshgrid(x, y)


    fig = plt.figure(figsize=(8,8))

    
    # Add a 3D subplot to the figure
    ax = fig.add_subplot(111)
 
    
    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    mag = np.sqrt(field[0]**2,field[1]**2)
    normalization= Normalize(vmin=0,vmax=0.5)
    surf = ax.quiver(X, Y, field[0],field[1],mag, cmap='viridis', scale=10.0,norm=normalization)
    
    # Set the title of the plot
    ax.set_title("Velocity Field - N = {} - Iteration {}".format(N,int(it)))
    
    # Save the figure
    plt.savefig('Frames/VectorFrames/' + name + "_" + str(int(it)) + '.png')
    
    # Close the figure to free memory
    plt.close(fig)


def plotPressureField(field, it, name):
    x = np.linspace(e0 - dH / 2, eF + dH / 2, N + 1)  # the line intervals
    y = np.linspace(e0 - dH / 2, eF + dH / 2, N + 1)  # the y intervals

    # Create a meshgrid for the surface plot
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define normalization for consistent color scaling
    normalization = Normalize(vmin=-0.4, vmax=0.4)

    # Plot a 3D surface with constant color scale
    surf = ax.plot_surface(X, Y, field, cmap='viridis', norm=normalization)

    # Set consistent z-axis limits
    ax.set_zlim(-0.4, 0.4)

    # Add a color bar with the same normalization
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Pressure ')

    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Pressure')
    ax.set_title("Pressure {} - Iteration {}".format(name,it))

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


#a = Lower Diag, b = Main Diag, c = Upper Diag, d = font
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


def tridiagonalX(size,vField,i):
    mainDiag = np.zeros(size)
    botton = np.zeros(size-1)
    top = np.zeros(size-1)

    #cctexacttion term

    #main diag is 1SIG
    mainDiag[:] = 1+SIG


    #botton diag is ro1/2 - sig/2, doest include the last term since its boundary
    botton[:] =    -(SIG/2) 

    #top diag is -ro1/2 - sig/2, doesnt include the first term since its boundary
    top[:] =   -(SIG/2) 


    mat = []

    mat.append(botton)
    mat.append(mainDiag)
    mat.append(top)

    return mat

def tridiagonalY(size, vField,collum):
    mainDiag = np.zeros(size)
    botton = np.zeros(size-1)
    top = np.zeros(size-1)

    #convection term


    mainDiag[:] = 1+SIG

    botton[:] =  -(SIG/2) 

    top[:] =    -(SIG/2) 


    mat = []
    mat.append(botton)   #0
    mat.append(mainDiag) #1
    mat.append(top)      #2

    return mat


def calculateDivergency(u,v):
    div = np.zeros((N-1,N-1))
    div = ((u[1:-1,1:] - u[1:-1,:-1]) / dH) + ((v[1:,1:-1] - v[:-1,1:-1])/ dH)
    return div

def calculateUPressureGradient(Pr):
    gPu = np.zeros((N+1,N))
    gPu[:,:] = (Pr[:,:-1] - Pr[:,1:]) / dH

    return gPu

def calculateVPressureGradient(Pr):
    gPv = np.zeros((N,N+1))
    gPv[:,:] = (Pr[:-1,:] - Pr[1:,:]) / dH

    return gPv

def plotDivergency(name,div,bdiv):
    ITs = np.linspace(0,len(div),len(div))
    plt.plot(ITs,bdiv,c="red",label="Divergency Before Correction")
    plt.plot(ITs,div,c="blue",label="Divergency After Correction")
    #plt.plot(ITs, np.cumsum(div), c="Red", label="Divergency Cumulative Sum")
    plt.title("Divergency Graph - N =  {}".format(N))
    plt.xlabel("Iteration") 
    plt.ylabel("Divergency Value")
    plt.legend()

    plt.savefig('Graphs/{}_N={}.png'.format(name,N))
    plt.close()
    return











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
u = exact(Xu,Yu,0)[0]
v = exact(Xv,Yv,0)[1]


#Pressure initial condition
P = np.zeros((N+1,N+1))
P = exactPressure(Xp,Yp,0)




#=========Solving ADI, assuming i = Y and j = X, Y growing donwards and X growing sideways, normally===========
#Initial conditions
uant = u
vant = v
Pant = P

#variables
umid = np.zeros((N+1,N))
vmid = np.zeros((N,N+1))
Pmid = np.zeros((N+1,N+1))

utio = np.zeros((N+1,N))
vtio = np.zeros((N,N+1))
#pressure grad

gradPu = calculateUPressureGradient(Pant)
gradPv = calculateVPressureGradient(Pant)

#velocity field, we need interpolation to solve the other component for each component, interpolateVtoU(vant) for example, interpolaates the V velocity component to the U positions
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


divAntes = []


matrixXu = tridiagonalX(N-2,bu,0)
matrixXv = tridiagonalX(N-1,bu,0)

matrixYu = tridiagonalY(N-1,bu,0)
matrixYv = tridiagonalY(N-2,bu,0)
itTime = []

BAR = tqdm(total=ITFINAL, desc="Progress")
while time < tF:

    BAR.update(1)

    t1 = process_time()

    #bounds conditions in all points
    umid[0,:] =   exact(xu,(e0 - dH/2) ,time)[0] #upper bounds
    umid[-1,:] =  exact(xu, (eF + dH/2),time)[0] #lower bounds
    umid[:,0] =   exact(e0,yu,time)[0]          #left side
    umid[:,-1] =  exact(eF,yu,time)[0]           #right side

    vmid[0,:] =   exact(xv,(e0) ,time)[1] #upper bounds
    vmid[-1,:] =  exact(xv,(eF) ,time)[1] #lower bonds
    vmid[:,0] =   exact(e0-(dH/2),yv ,time)[1] 
    vmid[:,-1] =  exact(eF+(dH/2),yv ,time)[1]

    u[0,:] =      exact(xu,(e0 - (dH/2)) ,time+ dT/2)[0] #upper bounds
    u[-1,:] =     exact(xu, (eF + (dH/2)),time+ dT/2)[0] #lower bounds
    u[:,0] =      exact(e0,yu,time+ dT/2)[0]          #left side
    u[:,-1] =     exact(eF,yu,time+ dT/2)[0]           #right side

    v[0,:] =      exact(xv,(e0) ,time+ dT/2)[1] #upper bounds
    v[-1,:] =     exact(xv,(eF) ,time+ dT/2)[1] #lower bonds
    v[:,0] =      exact(e0-(dH/2),yv ,time + dT/2)[1] 
    v[:,-1] =     exact(eF+(dH/2),yv ,time+ dT/2)[1]

    utio[0,:] =   exact(xu,(e0 - dH/2) ,time)[0] #upper bounds
    utio[-1,:] =  exact(xu, (eF + dH/2),time)[0] #lower bounds
    utio[:,0] =   exact(e0,yu,time)[0]          #left side
    utio[:,-1] =  exact(eF,yu,time)[0]           #right side

    vtio[0,:] =   exact(xv,(e0) ,time)[1] #upper bounds
    vtio[-1,:] =  exact(xv,(eF) ,time)[1] #lower bonds
    vtio[:,0] =   exact(e0-(dH/2),yv ,time)[1] 
    vtio[:,-1] =  exact(eF+(dH/2),yv ,time)[1]
    #testing with exact
    #vant = exact(Xv,Yv,time)[1]
    #vmid = exact(Xv,Yv,time+dT/2)[1]
    #v = exact(Xv,Yv,time+dT/2)[1]

    #uant = exact(Xu,Yu,time)[0]
    #umid = exact(Xu,Yu,time+dT/2)[0]
    #u = exact(Xu,Yu,time+dT/2)[0]


    #velocity field, for now, it is the exact solution at each point, we need to check how to do it
    bu = np.zeros((2,N+1,N))
    bv = np.zeros((2,N,N+1))
    bu[0] = uant
    bu[1] = interpolateVtoU(vant)
    bv[1] = vant#exact(Xv,Yv,time)[1]
    bv[0] = interpolateUtoV(uant)#exact(Xu,Yu,time)[0])

    roU1 = (bu[0]*dT/dH)[1:-1,1:-1]
    roU2 = (bu[1]*dT/dH)[1:-1,1:-1]

    roV1 = (bv[0]*dT/dH)[1:-1,1:-1]
    roV2 = (bv[1]*dT/dH)[1:-1,1:-1]

     #solve for the u~  and v~ 
    # we end up havingt o interpolate the vfield anyway
    # 
    #                                        //difusive stencil (4 cells), check the diagram                           c                    onvective term
    utio[1:-1,1:-1] = uant[1:-1,1:-1] + (SIG/2) * (uant[1:-1,:-2]+ uant[1:-1,2:]+  uant[2:,1:-1]+uant[:-2,1:-1]-4* uant[1:-1,1:-1]) - ((roU1/4)*(uant[1:-1,2:]- uant[1:-1,:-2]) + (roU2/4)*(uant[2:,1:-1]-uant[:-2,1:-1]))
    vtio[1:-1,1:-1] = vant[1:-1,1:-1] + (SIG/2) * (vant[1:-1,:-2]+ vant[1:-1,2:]+  vant[2:,1:-1]+vant[:-2,1:-1]-4* vant[1:-1,1:-1]) - ((roV1/4)*(vant[1:-1,2:]- vant[1:-1,:-2]) + (roV2/4)*(vant[2:,1:-1]-vant[:-2,1:-1]))


    


    #implicit in the x direction
    #for u, we need to go from one to N
    for i in range(1,N):
        ro2 = np.zeros(N) #velocity term
        ro1 = np.zeros(N)

        ro2[:] = (bu[1][i,:] * dT) /dH
        ro1[:] = (bu[0][i,:] * dT) /dH

        #font, we have the same number in x
        font = np.zeros(N-2)                                                                           #we need to add both of the terms here, but we have the problem with the counter
        font[:] = (1-SIG)*uant[i,1:-1] + ((SIG/2.0) * uant[i+1,1:-1])+ ((SIG/2.0) * uant[i-1,1:-1]) - (((ro1[1:-1]/4)*(utio[i,2:]- (utio[i,:-2])))  + (ro2[1:-1]/4)*(utio[i+1,1:-1]- (utio[i-1,1:-1]))) 
        font[0] +=  umid[i,0] *  (SIG/2)#border
        font[-1] += umid[i,-1] *  (SIG/2) #border


        umid[i,1:-1] = TDMA(matrixXu[0],matrixXu[1],matrixXu[2],font)
#      
    #for v, we have the usual limits
    for i in range(1,N-1):
        ro2 = np.zeros(N+1) #velocity term
        ro1 = np.zeros(N+1)

        ro2[:] = (bv[1][i,:] * dT) /dH
        ro1[:] = (bv[0][i,:] * dT) /dH

        #our font here is bigger because we have one more term
        font = np.zeros(N-1)
        font[:] = (1-SIG)*vant[i,1:-1] + ((SIG/2.0) * vant[i+1,1:-1])+ ((SIG/2.0) * vant[i-1,1:-1]) - (((ro1[1:-1]/4)*(vtio[i,2:]- (vtio[i,:-2])))  + (ro2[1:-1]/4)*(vtio[i+1,1:-1]- (vtio[i-1,1:-1]))) 
        font[0] +=  vmid[i,0] * ( SIG/2)#border
        font[-1] += vmid[i,-1]*( SIG/2) #border

        vmid[i,1:-1] = TDMA(matrixXv[0],matrixXv[1],matrixXv[2],font)






    #finished first half step, so we can now increase our time, and repeat the same process
    time = time+dT/2

    utio[0,:] =   exact(xu,(e0 - dH/2) ,time)[0] #upper bounds
    utio[-1,:] =  exact(xu, (eF + dH/2),time)[0] #lower bounds
    utio[:,0] =   exact(e0,yu,time)[0]          #left side
    utio[:,-1] =  exact(eF,yu,time)[0]           #right side

    vtio[0,:] =   exact(xv,(e0) ,time)[1] #upper bounds
    vtio[-1,:] =  exact(xv,(eF) ,time)[1] #lower bonds
    vtio[:,0] =   exact(e0-(dH/2),yv ,time)[1] 
    vtio[:,-1] =  exact(eF+(dH/2),yv ,time)[1]

    bu = np.zeros((2,N+1,N))
    bv = np.zeros((2,N,N+1))
    bu[0] = umid
    bu[1] = interpolateVtoU(vmid)
    bv[1] = vmid#exact(Xv,Yv,time)[1]
    bv[0] = interpolateUtoV(umid)#exact(Xu,Yu,time)[0])
    roU1 = (bu[0]*dT/dH)[1:-1,1:-1]
    roU2 = (bu[1]*dT/dH)[1:-1,1:-1]
    roV1 = (bv[0]*dT/dH)[1:-1,1:-1]
    roV2 = (bv[1]*dT/dH)[1:-1,1:-1]
    utio[1:-1,1:-1] = umid[1:-1,1:-1] + (SIG/2) * (umid[1:-1,:-2]+ umid[1:-1,2:]+  umid[2:,1:-1]+umid[:-2,1:-1]-4* umid[1:-1,1:-1]) - ((roU1/4)*(umid[1:-1,2:]- umid[1:-1,:-2]) + (roU2/4)*(umid[2:,1:-1]-umid[:-2,1:-1]))
    vtio[1:-1,1:-1] = vmid[1:-1,1:-1] + (SIG/2) * (vmid[1:-1,:-2]+ vmid[1:-1,2:]+  vmid[2:,1:-1]+vmid[:-2,1:-1]-4* vmid[1:-1,1:-1]) - ((roV1/4)*(vmid[1:-1,2:]- vmid[1:-1,:-2]) + (roV2/4)*(vmid[2:,1:-1]-vmid[:-2,1:-1]))



    #for the u component, in the y axis, we have the usual size
    for j in range(1,N-1):
        ro2 = np.zeros(N+1) #velocity term
        ro1 = np.zeros(N+1)
        ro1[:] = (bu[0][:, j] * dT)/dH
        ro2[:] = (bu[1][:, j] * dT)/dH

        #we have now a extra term in the font
        font = np.zeros(N-1)
        font[:] = (1-SIG)* umid[1:-1,j] + ((SIG/2.0)* umid[1:-1,j+1]) + ((SIG/2.0)* umid[1:-1,j-1]) - (((ro1[1:-1]/4)*(utio[1:-1,j+1]- (utio[1:-1,j-1])))  + (ro2[1:-1]/4)*(utio[2:,j]- (utio[:-2,j]))) 
        font[0] += u[0,j]* (SIG/2)
        font[-1] += u[-1,j]*(SIG/2) 


        u[1:-1,j] = TDMA(matrixYu[0],matrixYu[1],matrixYu[2],font)

    for j in range(1,N):
        ro2 = np.zeros(N) 
        ro1 = np.zeros(N)
        ro1[:] = (bv[0][:, j] * dT)/dH
        ro2[:] = (bv[1][:, j] * dT)/dH


        font = np.zeros(N-2)
        font[:] = (1-SIG)* vmid[1:-1,j] + ((SIG/2.0)* vmid[1:-1,j+1]) + ((SIG/2.0)* vmid[1:-1,j-1]) -  (((ro1[1:-1]/4)*(vtio[1:-1,j+1]- (vtio[1:-1,j-1])))  + (ro2[1:-1]/4)*(vtio[2:,j]- (vtio[:-2,j]))) 
        font[0] += v[0,j]  * (SIG/2)
        font[-1] += v[-1,j]*(SIG/2) 

        v[1:-1,j] = TDMA(matrixYv[0],matrixYv[1],matrixYv[2],font)

    #the algorithm has finished the step, so wwe increment the time and swap everything
    #also, we plot the solution, but we have to interpolate it back first
    time+= dT/2

    #solving the pressure now that we have a non-conservative velocity field
    #the procesure is based in the helmholtz decomposition and 
    #the relation between the pressure and velocity that arises when
    #we take the divergence of both sides in the
    #momentum equations - > we have -> laplacian P =  div u
    #whe should be able to solve for P (maybe we can even solve the for the gradient directly)

    t2 = process_time() 
    itTime.append(t2-t1)
    #solving P,using the linalg solver for now, we have a extra cell at every edge, because we need the grad, ask questions
    #solve p
    ##pressure solving, final half step
    ###border conditions, tecnically, it should work, so...
    divAntes.append(np.sum(np.abs(calculateDivergency(u,v))))
    B = calculateDivergency(u,v)
    P = np.zeros((N+1,N+1))
    P[0,:] = exactPressure(xp,(e0-dH/2),time-dT/2)
    P[-1,:] = exactPressure(xp,(eF+dH/2),time-dT/2)
    P[:,0] = exactPressure((e0-dH/2),yp,time-dT/2)
    P[:,-1]= exactPressure((eF+dH/2),yp,time-dT/2)
    Pant[0,:] = exactPressure(xp,(e0-dH/2),time-dT/2)
    Pant[-1,:] = exactPressure(xp,(eF+dH/2),time-dT/2)
    Pant[:,0] = exactPressure((e0-dH/2),yp,time-dT/2)
    Pant[:,-1]= exactPressure((eF+dH/2),yp,time-dT/2)
    itCount = 0

    while(True):
        itCount +=1
        for i in range(1, N):
            for j in range(1, N):
                P[i, j] =  (Pant[i-1, j]+Pant[i+1, j]+Pant[i, j-1] + Pant[i, j+1] + (B[i-1, j-1]*((dH**2)/(dT))))/(4)

        #update fields
        if(np.max(np.abs(P - Pant)) < 1e-12 and itCount > 8): #se a dif entre ambos for menor que -12, parar
            break
        Pant = P 


    #P = exactPressure(Xp,Yp,time-dT/2)


    gradPu = calculateUPressureGradient(P)
    gradPv = calculateVPressureGradient(P)
    ##subtract gradP from solution 
    u[1:-1,1:-1] = u[1:-1,1:-1] - (dT)*gradPu[1:-1,1:-1]
    v[1:-1,1:-1] = v[1:-1,1:-1] - (dT)*gradPv[1:-1,1:-1]



    uant = u
    vant = v
    umid = np.zeros((N+1,N))
    vmid = np.zeros((N,N+1))
    utio = np.zeros((N+1,N))
    vtio = np.zeros((N,N+1))
    u = np.zeros((N+1,N))
    v = np.zeros((N,N+1))

    Pant = P
    P = np.zeros((N+1,N+1))

    #error calculation, we calculate it before the interpolation, in each component



    gradError.append(np.max(gradPu -exactGradP(Xu,Yu,time-dT/2)[0]))
    gradError.append(np.max(gradPv -exactGradP(Xv,Yv,time-dT/2)[1]))

    uexact = exact(Xu,Yu,time-dT/2)[0] #we are half a dt foward, so we just subtract it 
    vexact = exact(Xv,Yv,time-dT/2)[1] #we are half a dt foward, so we just subtract it 
    error.append(np.max(np.abs(uexact - uant)))
    error.append(np.max(np.abs(vexact - vant)))
    pError.append(np.max(Pant - exactPressure(Xp,Yp,time-dT/2)))

    div.append(np.sum(np.abs(calculateDivergency(uant,vant)))) #talvez dividir por N






    #Plotting, uncomment each plot separately if you wanna actually run this, this is very heavy
    #plot the aproximate and the exact solution just for comparison, interpolated to the same point
    #U = interpolate(uant,vant)
    #plotVectorField(U,IT,"Interpolated")
    #U = interpolate(uexact,vexact)
    #plotVectorField(U,IT,"InterpolatedExact")
    #####and also plot the individual points
    #plotScalarFieldX(uant,IT,"ScalarU")
    #plotScalarFieldX(uexact,IT,"ScalarUExact")
    #####and also the pressure gradient interpolated
    ####PressureInterpolated = interpolate(gradPu,gradPv)
    ####plotVectorField(PressureInterpolated,IT,"Pressure_Gradient")
    ###PressureInterpolated = interpolate(gradP(Xu,Yu,time)[0],gradP(Xv,Yv,time)[1])
    ###plotVectorField(PressureInterpolated,IT,"Pressure_Gradient_Exact")
    #plotPressureField(exactPressure(Xp,Yp,time),IT,"Exact")
    #plotPressureField(Pant,IT,"Aproximate - N = {}".format(N))

    

    IT +=1


##after the main loop is finished, we just take the maximum of the both components max error in each interaction
with open("Error/time_explicit.csv", 'a') as file:
    file.write(f'{N},{np.sum(itTime)/len(itTime)}\n')

with open("Error/error.csv", 'a') as file:
    file.write(f'{N},{max(error)}\n')

#with open("Error/errorGrad.csv", 'a') as file:
#    file.write(f'{N},{max(gradError)}\n')
# 
#with open("Error/errorPressure.csv", 'a') as file:
#    file.write(f'{N},{max(pError)}\n')
##plot the divergency graph
#plotDivergency("Taylor Green - SE",div,divAntes)

