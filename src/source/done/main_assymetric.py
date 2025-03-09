import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import Normalize
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
#What is left...
#Make the domain assymetic, longer on X, specifically
from time import process_time



#Grid size


H = 1.0
DELTA =  0.5*H
U0 = 1.0

STEP_HEIGHT = 0.5*H
STEP_WIDTH = 6*H


#Grid size
N  = int(sys.argv[1])

Nx = 15*(N - 1) +1
Ny = N


#Time Domain
t0 = 0
tF = 0.5
#Space Domain
e0 = 0
eF =  1*H 



e0x = 0
eFx = 15*H

#Solver info
dH = (eF - e0) / (N-1)  

dT = (dH)/100
    

#Equations parameters
RE = 100
EPS = (2/3)/RE
SIG = (EPS*dT)/(dH**2)






print(dT)


def f(x,y,t):
    return 0


def stepBoundaryPressure(p, xp, yp):
    # Create the boolean mask
    mask = np.outer(yp < STEP_HEIGHT, xp < STEP_WIDTH)


    p[mask] = 0
    
    # force the edge
    i = int(np.floor(STEP_HEIGHT/dH)) +1 
    j = int(np.floor(STEP_WIDTH/dH)) +1


    p[i,:j] = p[i+1,:j] #top
    p[:i,j] = p[:i,j+1]#p[:i,j+1]  #right edge
    p[:i,0] = p[:i,1] #left dge
    p[0,:j] = p[1,:j]  #botton step edge
    #p[i,j] = (p[i,j+1] + p[i,j-1] + p[i+1,j] + p[i-1,j] )/4

    
    return p

def stepBoundaryU(u, xu, yu):
    # Create a boolean mask for the condition
    i = int(np.floor(STEP_HEIGHT/dH))  +1
    j = int(np.floor(STEP_WIDTH/dH))   +1
    mask = np.outer(yu < STEP_HEIGHT, xu < STEP_WIDTH)
    u[mask] = 0

    u[i,:j] = 0#u[i+1,:j] #top edge
    u[:i,j] = 0 #right edge
    u[:i,0] = 0 #left dge
    u[0,:j] = 0  #botton step edge

    return u
 
def stepBoundaryV(v, xv, yv):
    # Create a boolean mask for the condition
    mask = np.outer(yv < STEP_HEIGHT, xv < STEP_WIDTH)
    v[mask] = 0

    i = int(np.floor(STEP_HEIGHT/dH))  +1
    j = int(np.floor(STEP_WIDTH/dH))   +1
    v[i,:j] = 0 #top edge
    v[:i,j] = 0#v[:i,j+1] #right edge
    v[:i,0] = 0 #left dge
    v[0,:j] = 0  #botton step edge
    return v




def lefBoundary(y):
    #return np.where(y >= STEP_HEIGHT, U0, U0)

    return np.where(y > STEP_HEIGHT, -(y - (3*STEP_HEIGHT/2)  )**(2) + U0, 0)
                                             #true  #false
    #return np.where(y >= DELTA + STEP_HEIGHT, U0, ((y / DELTA) ** (0.14285714285) )* U0)
    





import numpy as np
import matplotlib.pyplot as plt

def plotVectorField(field, it, name):
    # Offset the cell centers by dH/2
    x = np.linspace(e0x + dH / 2, eFx - dH / 2, Nx - 1)
    y = np.linspace(e0 + dH / 2, eF - dH / 2, N - 1)
    X, Y = np.meshgrid(x, y)

    # Create the figure
    fig = plt.figure(figsize=(32, 10))

    # First subplot: imshow of magnitude
    ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.4])  # [left, bottom, width, height]
    mag = np.sqrt(field[0] ** 2 + field[1] ** 2)
    img = ax1.imshow(
        mag, extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower', cmap='viridis', interpolation='bilinear'
    )
    ax1.set_ylim((0, 1.0))
    ax1.set_ylabel("y")
    ax1.set_title(f"Velocity Magnitude - N = {N} - Iteration {int(it)}")
    ax1.set_aspect('equal')  # Match aspect ratio

    cbar = fig.colorbar(img, ax=ax1, shrink=0.5, aspect=5,orientation='horizontal')
    cbar.set_label('Magnitude')

    # Second subplot: streamlines only
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])  # [left, bottom, width, height]
    seed_x = np.linspace(x[1], x[-2], 10)  # Adjust number of seeds as needed
    seed_y = np.linspace(y[1], y[-2], 10)
    seed_points = np.array([(sx, sy) for sx in seed_x for sy in seed_y])


    ax2.streamplot(X, Y, field[0], field[1], color="black", start_points=seed_points,linewidth=0.1,density=6.0,arrowsize=0.00001 )
    ax2.set_ylim((0, 1.0))
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Streamlines - N = {} - Iteration {}".format(N, int(it)))
    ax2.set_aspect('equal')  # Match aspect ratio
    # Set major and minor ticks
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # Major ticks with integer values
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))  # Minor ticks: you can adjust the number of subticks

    # Show minor ticks
    ax2.tick_params(which='both', axis='x', direction='in', length=6)
    ax2.tick_params(which='minor', axis='x', length=3) 

    # Save the figure
    plt.savefig('Frames/VectorFrames/' + name + "_" + str(int(it)) + '.png')

    # Close the figure to free memory
    plt.close(fig)



def plotPressureField(field, it, name):
    x = np.linspace((e0x - dH / 2), (eFx + dH / 2), Nx + 1)
    x = x / H
    y = np.linspace((e0 - dH / 2), (eF + dH / 2), N + 1)
    X, Y = np.meshgrid(x, y)

    # Create the figure and define two subplots
    fig = plt.figure(figsize=(32, 8))  # Adjust figure size to accommodate both plots
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)  # GridSpec for layout

    # Main plot (Pressure Field)
    ax_main = fig.add_subplot(gs[0])
    mask = np.zeros_like(field, dtype=bool)
    i_end = int(np.floor(STEP_HEIGHT / dH)) + 1
    j_end = int(np.floor(STEP_WIDTH / dH)) + 1
    mask[0:i_end, 0:j_end] = True  # Define the region to mask
    masked_field = np.ma.array(field, mask=mask)
    j = int((STEP_WIDTH) // dH) + 1
    i = int(STEP_HEIGHT // dH) + 1
    surf = ax_main.pcolormesh(X, Y, (masked_field) /  (U0 ** 2), cmap='viridis')

    # Add color bar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Add color bar next to the plot
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label('Pressure')

    # Labels and title
    ax_main.set_xlabel('x/H')
    ax_main.set_ylabel('y')
    ax_main.set_title("Cₚ Field")

    # Bottom plot (Cross-sectional cut along x-axis at a fixed y location)
    ax_bottom = fig.add_subplot(gs[1])
    y_idx = i  # Select a specific y position (middle of the domain, modify as needed)
    x_cut = x  # x values
    pressure_cut = (field[y_idx, :] - field[i, j]) / ((1 / 2) * (U0 ** 2))  # Pressure values along x-axis

    ax_bottom.plot(x_cut, pressure_cut)
    ax_bottom.set_xlabel('x/H')
    ax_bottom.set_ylabel('Normalized Pressure')
    ax_bottom.set_title('Cross-sectional Cₚ at Step Height')
    ax_bottom.legend()

    # Save the figure
    plt.savefig(f'Frames/ScalarFrames/Pressure/{name}_{int(it)}.png')

    # Close
    plt.close()

def plotUScalarField(field, it, name):
    x = np.linspace(e0x, eFx, Nx) #the line intervals
    y = np.linspace(e0-(dH/2), (eF+dH/2),N+1)
    X, Y = np.meshgrid(xu  , yu) #the complete grid
    X, Y = np.meshgrid(x, y)

    # Create the figure and define two subplots
    fig = plt.figure(figsize=(32, 8))  # Adjust figure size to accommodate both plots
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)  # GridSpec for layout

    # Main plot (Pressure Field)
    ax_main = fig.add_subplot(gs[0])
    mask = np.zeros_like(field, dtype=bool)
    i_end = int(np.floor(STEP_HEIGHT / dH)) + 1
    j_end = int(np.floor(STEP_WIDTH / dH)) + 1
    mask[0:0, 0:0] = True  # Define the region to mask
    masked_field = np.ma.array(field, mask=mask)
    j = int((STEP_WIDTH) // dH) + 1
    i = int(STEP_HEIGHT // dH) + 1
    surf = ax_main.pcolormesh(X, Y, (masked_field) /  (U0 ** 2), cmap='viridis')

    # Add color bar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Add color bar next to the plot
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label('Pressure')

    # Labels and title
    ax_main.set_xlabel('x/H')
    ax_main.set_ylabel('y')
    ax_main.set_title("{} Velocity Field".format())

    # Save the figure
    plt.savefig(f'Frames/ScalarFrames/U/{name}_{int(it)}.png')

    # Close
    plt.close()

#returns to the centered grid layout, receives the u and v stagerred components
def interpolate(u,v):
    U = np.zeros((2,N-1,Nx-1))
    U[0][:,:] = (u[1:-1,1:] +u[1:-1,:-1])/2.0   #+u[1:-1,2:-1])/2.0 # for each cell center, we interpolate the left and right edge for the u, and top and botton for v, very simple but easy to make mistakes with the indexes
    U[1][:,:] = (v[1:,1:-1] +v[:-1,1:-1])/2.0
    return U

#interpolates the V velocity nodes to the U nodes, we dont interpolate the outer nodes so we just let them in 0, theyre not used anyway
def interpolateVtoU(v):
    vU = np.zeros((N+1,Nx))
                    #Top 2 nodes            #botton 2
    vU[1:-1,:] = (((v[:-1,:-1] + v[:-1,1:])/2) + ((v[1:,:-1] + v[1:,1:])/2))/2
    return vU
#interpolates the U velocity nodes to the V nodes
def interpolateUtoV(u):
    uV = np.zeros((N,Nx+1))
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
    div = np.zeros((N-1,Nx-1))
    div = ((u[1:-1,1:] - u[1:-1,:-1]) / dH) + ((v[1:,1:-1] - v[:-1,1:-1])/ dH)
    return div

def calculateUPressureGradient(Pr):
    gPu = np.zeros((N+1,Nx))
    gPu[:,:] = (Pr[:,:-1] - Pr[:,1:]) / dH

    return gPu

def calculateVPressureGradient(Pr):
    gPv = np.zeros((N,Nx+1))
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




animeationFrame = 0

# ---------------> SMALL NOTATION COMMENT <---------------
#we use u and v for the vector components, these are in the cells edges
#we use U for the whole vector, this is in the cell center, interpolated back from the edges
#we use x,y for the axis, or X,Y for the whole numpy grid
#our y increases with i




#solving the staggered grid in their points, check notes, this is VERY delicate work, ---> (CHECK THE DIAGRAM, WHICH SHOULD BE IN THE COLAB ATTACHMENT) <----
#note that, in u, there is a extra element in the y axis, both downards and upwards, because of the grid aligment

#in U, our grid is descolated half of dh downards (or upwards, depends in your index definitions) in relation to cell centers
#discrete U grid
xu = np.linspace(e0x, eFx, Nx) #the line intervals
yu = np.linspace(e0-(dH/2), (eF+dH/2),N+1)
Xu, Yu = np.meshgrid(xu  , yu) #the complete grid



#exactly the same thing but, for the v, we dislocate in the x axis
xv = np.linspace(e0x-(dH/2), (eFx+dH/2),Nx+1) 
yv =  np.linspace(e0, eF, N) 
Xv, Yv = np.meshgrid(xv,yv)



#discrete Pressure grid
xp = np.linspace((e0x - dH/2),(eFx + dH/2),Nx+1)
yp = np.linspace((e0 - dH/2),(eF + dH/2),N+1)
Xp,Yp = np.meshgrid(xp,yp)


#solve the initial condition for u and v, interpolate back to the U 
#            i(y),j(x)
u = np.zeros((N+1,Nx))
v = np.zeros((N,Nx+1))



#Pressure initial condition
P = np.zeros((N+1,Nx+1))






#=========Solving ADI, assuming i = Y and j = X, Y growing donwards and X growing sideways, normally===========
#Initial conditions
uant = u
vant = v
Pant = P

#variables
umid = np.zeros((N+1,Nx))
vmid = np.zeros((N,Nx+1))
Pmid = np.zeros((N+1,Nx+1))

#pressure grad

gradPu = calculateUPressureGradient(Pant)
gradPv = calculateVPressureGradient(Pant)

#velocity field, we need interpolation to solve the other component for each component, interpolateVtoU(vant) for example, interpolaates the V velocity component to the U positions
bu = np.zeros((2,N+1,Nx))
bv = np.zeros((2,N,Nx+1))
bu[0] = uant
bu[1] = interpolateVtoU(vant)
bv[0] = vant
bv[1] = interpolateUtoV(uant)

U = interpolate(u,v)
plotVectorField(U,0,"Interpolated")

#plotPressureField(P,0,"Pressure")
#plotPressureField(Pant,0,"Pressure_Exact")


#control variables
IT = 1

print(xu[1] - xu[0])
print(yu[1] - yu[0])


ITFINAL = ((tF-dT)/dT)
time = 0
time = time+dT/2 

#error variabes

gradError = []
error = []
pError = []
div = []


divAntes = []

itTime = []



def plotBoundCondition(upar):
    plt.plot(upar,yu)
    plt.title("U Boundary Condition Entry")
    plt.xlabel("u component")
    plt.ylabel("y")
    plt.show()

BAR = tqdm(total=ITFINAL, desc="Progress")
stable = False
while stable == False:





    #bounds conditions in all points
    umid[0,:] =   0#exact(xu,(e0 - dH/2) ,time)[0] #lower
    umid[-1,:] =  0#exact(xu, (eF + dH/2),time)[0] #upper
    umid[:,0] =   lefBoundary(yu)#exact(e0x,yu,time)[0]          #left
    umid[:,-1] =  umid[:,-2]#exact(eFx,yu,time)[0]          #right    


    vmid[0,:] =   0#exact(xv,(e0) ,time)[1] #              #lower
    vmid[-1,:] =  0#exact(xv,(eF) ,time)[1] #              #upper 
    vmid[:,0] =   0#exact(e0x-(dH/2),yv ,time)[1]          #left
    vmid[:,-1] =  vmid[:,-2]#exact(eFx+(dH/2),yv ,time)[1]          #right

    u[0,:] =      0#exact(xu,(e0 - (dH/2)) ,time+ dT/2)[0] #lower
    u[-1,:] =     0 #exact(xu, (eF + (dH/2)),time+ dT/2)[0] #upper
    u[:,0] =      lefBoundary(yu) #exact(e0x,yu,time+ dT/2)[0]            #left
    u[:,-1] =     u[:,-2]#exact(eFx,yu,time+ dT/2)[0]            #right

    v[0,:] =      0#exact(xv,(e0) ,time+ dT/2)[1]         #lower
    v[-1,:] =     0#exact(xv,(eF) ,time+ dT/2)[1]         #upper
    v[:,0] =      0#exact(e0x-(dH/2),yv ,time + dT/2)[1]  #left
    v[:,-1] =     v[:,-2] #  0#exact(eFx+(dH/2),yv ,time+ dT/2)[1]   #right


    v = stepBoundaryV(v,xv,yv)
    vmid = stepBoundaryV(vmid,xv,yv)
    u = stepBoundaryU(u,xu,yu)
    umid = stepBoundaryU(umid,xu,yu)

    #plotBoundCondition(u[:,0])





    

    #print("P_width: ",xp[1] - xp[0])
    #print("P_height: ",yp[1] - yp[0])
    #print("WIDTH: ",STEP_WIDTH)
    #print("HEIGHT: ",STEP_HEIGHT)
 



    


    #velocity field, for now, it is the exact solution at each point, we need to check how to do it
    bu = np.zeros((2,N+1,Nx))
    bv = np.zeros((2,N,Nx+1))
    bu[0] = uant#exact(Xu,Yu,time)[0]
    bu[1] = interpolateVtoU(vant)#exact(Xv,Yv,time)[1])


    bv[1] = vant#exact(Xv,Yv,time)[1]
    bv[0] = interpolateUtoV(uant)#exact(Xu,Yu,time)[0])


    #implicit in the x direction
    #for u, we need to go from one to N
    for i in range(1,N):
        ro2 = np.zeros(Nx) #velocity term
        ro1 = np.zeros(Nx)

        ro2[:] = (bu[1][i,:] * dT) /dH
        ro1[:] = (bu[0][i,:] * dT) /dH

        #font, we have the same number in x
        font = np.zeros(Nx-2) 
        font[:] = (1-SIG)*uant[i,1:-1] + ((SIG/2.0 - ro2[1:-1]/4) * uant[i+1,1:-1])+ ((SIG/2.0 + ro2[1:-1]/4) * uant[i-1,1:-1]) 
        font[0] +=  umid[i,0] *  (ro1[1]/4 + SIG/2)#border
        font[-1] += umid[i,-1] *  (-ro1[-2]/4 + SIG/2) #border

        matrix = tridiagonalX(Nx-2,bu,i)
        umid[i,1:-1] = TDMA(matrix[0],matrix[1],matrix[2],font)

    #for v, we have the usual limits
    for i in range(1,N-1):
        ro2 = np.zeros(Nx+1) #velocity term
        ro1 = np.zeros(Nx+1)

        ro2[:] = (bv[1][i,:] * dT) /dH
        ro1[:] = (bv[0][i,:] * dT) /dH

        #our font here is bigger because we have one more term
        font = np.zeros(Nx-1)
        font[:] = (1-SIG)*vant[i,1:-1] + ((SIG/2.0 - ro2[1:-1]/4) * vant[i+1,1:-1])+ ((SIG/2.0 + ro2[1:-1]/4) * vant[i-1,1:-1]) 
        font[0] +=  vmid[i,0] * (ro1[1]/4 + SIG/2)#border
        font[-1] += vmid[i,-1]*(-ro1[-2]/4 + SIG/2) #border

        matrix = tridiagonalX(Nx-1,bv,i)
        vmid[i,1:-1] = TDMA(matrix[0],matrix[1],matrix[2],font)





    #finished first half step, so we can now increase our time, and repeat the same process
    time = time+dT/2

    bu = np.zeros((2,N+1,Nx))
    bv = np.zeros((2,N,Nx+1))
    bu[0] = umid#exact(Xu,Yu,time)[0]
    bu[1] = interpolateVtoU(vmid)#exact(Xv,Yv,time)[1])


    bv[1] = vmid#exact(Xv,Yv,time)[1]
    bv[0] = interpolateUtoV(umid)#exact(Xu,Yu,time)[0])

    v = stepBoundaryV(v,xv,yv)
    vmid = stepBoundaryV(vmid,xv,yv)
    u = stepBoundaryU(u,xu,yu)
    umid = stepBoundaryU(umid,xu,yu)

    

    #for the u component, in the y axis, we have the usual size
    for j in range(1,Nx-1):
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

    for j in range(1,Nx):
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
    v = stepBoundaryV(v,xv,yv)
    vmid = stepBoundaryV(vmid,xv,yv)
    u = stepBoundaryU(u,xu,yu)
    umid = stepBoundaryU(umid,xu,yu)
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

    divAntes.append(np.sum(np.abs(calculateDivergency(u,v))))
    B = calculateDivergency(u,v)
    P = np.zeros((N+1,Nx+1))

    P = stepBoundaryPressure(P,xp,yp)
    Pant = stepBoundaryPressure(Pant,xp,yp)
    Pant[:,0] =     Pant[:,1]  
    Pant[:,-1] =    Pant[:,-2]+ (dH/dT)*(u[:,-2] )
    Pant[0,:] =     Pant[1,:]   #+ (dH/dT)*( v[0,:]-v[1,:])
    Pant[-1,:] =    Pant[-2,:]
    P[:,0] =     P[:,1]   
    P[:,-1] =    P[:,-2]+ (dH/dT)*(u[:,-2])
    P[0,:] =     P[1,:]   #+ (dH/dT)*( v[0,:]-v[1,:])
    P[-1,:] =    P[-2,:]
    P = stepBoundaryPressure(P,xp,yp)
    Pant = stepBoundaryPressure(Pant,xp,yp)





    itCount = 0

    while(True):
        P = stepBoundaryPressure(P,xp,yp)
        Pant = stepBoundaryPressure(Pant,xp,yp)
        Pant[:,0] =     Pant[:,1]  
        Pant[:,-1] =     Pant[:,-2]+ (dH/dT)*(u[:,-2] )
        Pant[0,:] =     Pant[1,:]   #+ (dH/dT)*( v[0,:]-v[1,:])
        Pant[-1,:] =    Pant[-2,:]

        P[:,0] =     P[:,1]   
        P[:,-1] =    P[:,-2]+ (dH/dT)*(u[:,-2])
        P[0,:] =     P[1,:]   #+ (dH/dT)*( v[0,:]-v[1,:])
        P[-1,:] =    P[-2,:]
        P = stepBoundaryPressure(P,xp,yp)
        Pant = stepBoundaryPressure(Pant,xp,yp)
        itCount +=1
        for i in range(1, N):
            for j in range(1, Nx):
                P[i, j] =  (Pant[i-1, j]+Pant[i+1, j]+Pant[i, j-1] + Pant[i, j+1] + (B[i-1, j-1]*((dH**2)/(dT))))/(4)



        #update fields
        if(np.max(np.abs(P - Pant)) < 1e-6 and itCount > 8): #se a dif entre ambos for menor que -12, parar
            break
        Pant = P 


    #P = exactPressure(Xp,Yp,time-dT/2)


    gradPu = calculateUPressureGradient(P)
    gradPv = calculateVPressureGradient(P)
    v = stepBoundaryV(v,xv,yv)
    vmid = stepBoundaryV(vmid,xv,yv)
    u = stepBoundaryU(u,xu,yu)
    umid = stepBoundaryU(umid,xu,yu)
    ##subtract gradP from solution 
    u[1:-1,1:-1] = u[1:-1,1:-1] - (dT)*gradPu[1:-1,1:-1]
    v[1:-1,1:-1] = v[1:-1,1:-1] - (dT)*gradPv[1:-1,1:-1]

    v = stepBoundaryV(v,xv,yv)
    vmid = stepBoundaryV(vmid,xv,yv)
    u = stepBoundaryU(u,xu,yu)
    umid = stepBoundaryU(umid,xu,yu)

    u[0,:] =      0#exact(xu,(e0 - (dH/2)) ,time+ dT/2)[0] #lower
    u[-1,:] =     0 #exact(xu, (eF + (dH/2)),time+ dT/2)[0] #upper
    u[:,0] =      lefBoundary(yu) #exact(e0x,yu,time+ dT/2)[0]            #left
    u[:,-1] =     u[:,-2]#exact(eFx,yu,time+ dT/2)[0]            #right

    v[0,:] =      0#exact(xv,(e0) ,time+ dT/2)[1]         #lower
    v[-1,:] =     0#exact(xv,(eF) ,time+ dT/2)[1]         #upper
    v[:,0] =      0#exact(e0x-(dH/2),yv ,time + dT/2)[1]  #left
    v[:,-1] =     v[:,-2] #  0#exact(eFx+(dH/2),yv ,time+ dT/2)[1]   #right




    Uant = interpolate(uant,vant)
    U = interpolate(u,v)
    diff = np.abs(U - Uant)
    print("Pressure converged in {} iterations --- Dif {} -- It = {} -- t = {}".format(itCount,np.max(diff),IT,time),end="\r")
    if(np.max(diff) < 1e-12 and IT > 1000): 
        print("SIMULATION STABLE -- ENDING")
        stable = True

    uant = u
    vant = v

    umid = np.zeros((N+1,Nx))
    vmid = np.zeros((N,Nx+1))
    u = np.zeros((N+1,Nx))
    v = np.zeros((N,Nx+1))

    Pant = P
    P = np.zeros((N+1,Nx+1))
    P = stepBoundaryPressure(P,xp,yp)
    Pant = stepBoundaryPressure(Pant,xp,yp)
    Pant[:,0] =     Pant[:,1]  
    Pant[:,-1] =    Pant[:,-2]+ (dH/dT)*(u[:,-2] )
    Pant[0,:] =     Pant[1,:]   #+ (dH/dT)*( v[0,:]-v[1,:])
    Pant[-1,:] =    Pant[-2,:]
    P[:,0] =     P[:,1]   
    P[:,-1] =    P[:,-2]+ (dH/dT)*(u[:,-2])
    P[0,:] =     P[1,:]   #+ (dH/dT)*( v[0,:]-v[1,:])
    P[-1,:] =    P[-2,:]
    P = stepBoundaryPressure(P,xp,yp)
    Pant = stepBoundaryPressure(Pant,xp,yp)

    v = stepBoundaryV(v,xv,yv)
    vmid = stepBoundaryV(vmid,xv,yv)
    u = stepBoundaryU(u,xu,yu)
    umid = stepBoundaryU(umid,xu,yu)



    #error calculation, we calculate it before the interpolation, in each component


    div.append(np.sum(np.abs(calculateDivergency(uant,vant)[1:-1,1:-1]))) #talvez dividir por N
    #sample size





    if(IT%10 == 0):
        #Plotting, uncomment each plot separately if you wanna actually run this, this is very heavy
        #plot the aproximate and the exact solution just for comparison, interpolated to the same point
        U = interpolate(uant,vant)
        plotVectorField(U,animeationFrame,"Interpolated")

        #U = interpolate(uexact,vexact)
        #plotVectorField(U,IT,"InterpolatedExact")
        #####and also plot the individual points
        ####plotScalarFieldX(uant,IT,"ScalarU")
        ####plotScalarFieldX(uexact,IT,"ScalarUExact")
        #####and also the pressure gradient interpolated
        ####PressureInterpolated = interpolate(gradPu,gradPv)
        ####plotVectorField(PressureInterpolated,IT,"Pressure_Gradient")
        ###PressureInterpolated = interpolate(gradP(Xu,Yu,time)[0],gradP(Xv,Yv,time)[1])
        ###plotVectorField(PressureInterpolated,IT,"Pressure_Gradient_Exact")
        #plotPressureField(exactPressure(Xp,Yp,time),IT,"Exact")
        plotPressureField(Pant,animeationFrame,"Pressure")
        animeationFrame +=1
    if(IT > 10000):
        break





    IT +=1


##after the main loop is finished, we just take the maximum of the both components max error in each interaction

#
#with open("Error/errorGrad.csv", 'a') as file:
#    file.write(f'{N},{max(gradError)}\n')
# 
#with open("Error/errorPressure.csv", 'a') as file:
#    file.write(f'{N},{max(pError)}\n')

#with open("Error/time.csv", 'a') as file:
#    file.write(f'{N},{np.sum(itTime)/len(itTime)}\n')
#plot the divergency graph
plotDivergency("Backwards Facing Step",div,divAntes)




