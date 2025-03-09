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
from time import perf_counter
from time import perf_counter

import ray


import pandas as pd


from concurrent.futures import ProcessPoolExecutor
import multiprocessing


#Grid size


ray.init()

X_SIZE = 30
Y_SIZE = 1
H = 1.0
DELTA =  0.5*H
U0 = 1.0
STEP_HEIGHT = 0.5*H
STEP_WIDTH = 0.5*H




#Grid size
N  = int(sys.argv[1])

Nx = X_SIZE*(N - 1) +1
Ny = N


#Time Domain
t0 = 0
tF = 0.5
#Space Domain
e0 = 0
eF =  1*H 



e0x = 0
eFx = X_SIZE*eF

#Solver info
dH = (eF - e0) / (N-1)  

dT = (dH/N)
    

#Equations parameters
RE = 600
EPS = (2/3)/RE
SIG = (EPS*dT)/(dH**2)




STEP_I = int(np.floor(STEP_HEIGHT/dH))  +1
STEP_J = int(np.floor(STEP_WIDTH/dH))   +1




def f(x,y,t):
    return 0


def stepBoundaryPressure(p, xp, yp,uarg,varg,uanterior,vanterior):
    # Create the boolean mask
    mask = np.outer(yp < STEP_HEIGHT, xp < STEP_WIDTH)
    p_r = p.copy()

    p_r[mask] = 0

    p_r[:,0] =     p[:,1]  
    p_r[0,:] =     p[1,:] 
    p_r[-1,:] =    p[-2,:]
    p_r[:,-1]  =   p[:,-2] 

    #_r force the edge
    p_r[STEP_I,:STEP_J] = p[STEP_I+1,:STEP_J] #top
    p_r[:STEP_I,STEP_J] = p[:STEP_I,STEP_J+1] #p[:i,j+1]  #right edge
    p_r[:STEP_I,0] = p[:STEP_I,1] #left dge
    p_r[0,:STEP_J] = p[1,:STEP_J]  #botton step edge
    p_r[STEP_I,STEP_J] = (p[STEP_I,STEP_J+1] + p[STEP_I,STEP_J-1] + p[STEP_I+1,STEP_J] + p[STEP_I-1,STEP_J] )/4


    
    return p

def stepBoundaryU(uarg, xuarg, yuarg):

    uarg_r = uarg.copy()

    uarg_r[:,0] =      lefBoundary(yuarg) #exact(e0x,yu,time+ dT/2)[0]            #left
    uarg_r[-1,:] =     0#uarg[-2,:]
    uarg_r[0,:] =      0#exact(xu,(e0 - (dH/2)) ,time+ dT/2)[0] #lower
    uarg_r[:,-1] =     uarg[:,-2]#exact(eFx,yu,time+ dT/2)[0]            #right

    # Cr_reate a boolean mask for the condition
    mask = np.outer(yuarg < STEP_HEIGHT+(dH/2), xuarg < STEP_WIDTH+(dH/2))
    uarg_r[mask] = 0

    uarg_r[:STEP_I,0] = 0 #left dge
    uarg_r[0,:STEP_J] = 0  #botton step edge

    uarg_r[STEP_I,:STEP_J+1] = 0#uarg[STEP_I+1,:STEP_J+1] # top
    uarg_r[:STEP_I,STEP_J] = 0#uarg[:STEP_I,STEP_J+1] #right edge
    #uarg[STEP_I,STEP_J] =  0#uarg[STEP_I,STEP_J+1] #corner




    return uarg_r

def stepBoundaryV(varg, xvarg, yvarg):
    # Create a boolean mask for the condition

    varg_r = varg.copy()


    varg_r[:,0] =      0#exact(e0x-(dH/2),yv ,time + dT/2)[1]  #left
    varg_r[-1,:] =     0 #0#exact(xv,(eF) ,time+ dT/2)[1]         #upper
    varg_r[0,:] =      0#exact(xv,(e0) ,time+ dT/2)[1]         #lower
    varg_r[:,-1] =     varg[:,-2] #  0#exact(eFx+(dH/2),yv ,time+ dT/2)[1]   #right

    mask = np.outer(yvarg < STEP_HEIGHT+(dH/2), xvarg < STEP_WIDTH+(dH/2))
    varg_r[mask] = 0
    varg_r[:STEP_I,STEP_J] = 0#v[:i,j+1] #right edge
    varg_r[:STEP_I,0] = 0 #left dge
    varg_r[0,:STEP_J] = 0  #botton step edge

    varg_r[:STEP_I+1,STEP_J] = 0#varg[:STEP_I+1,STEP_J+1] #right edge
    varg_r[STEP_I,:STEP_J] = 0  #top edge
    #varg[STEP_I,STEP_J] =  0#varg[STEP_I+1,STEP_J] #corner


    return varg_r




def lefBoundary(y):
    return np.where(y > STEP_HEIGHT, -16*(y - (3*STEP_HEIGHT/2)  )**(2) + U0, 0)



def plotVectorField(field, it, name, x_min=0, x_max=None):
    # Set global font size
    plt.rcParams.update({'font.size': 8})  # Increase font size for all elements

    # Offset the cell centers by dH/2
    x = np.linspace(e0x + dH / 2, eFx - dH / 2, Nx - 1)
    y = np.linspace(e0 + dH / 2, eF - dH / 2, N - 1)
    X, Y = np.meshgrid(x, y)
    jFinal = int(x_max/dH)

    # Determine the range for the X-axis
    if x_max is None:
        x_max = X.max()
    x_range = (x_min, x_max)

    # Create the figure
    fig = plt.figure(figsize=(x_max + 2 ,4))

    # First subplot: imshow of magnitude
    ax1 = fig.add_axes([0.1, 0.50, 0.8, 0.4])  # [left, bottom, width, height] - Adjusted bottom to 0.50
    mag = np.sqrt(field[0] ** 2 + field[1] ** 2)
    X = X[:, :jFinal]
    Y = Y[:, :jFinal]

    # Apply the X-axis limits to the imshow plot
    img = ax1.imshow(
        mag[:, :jFinal], extent=[x_range[0], x_range[1], Y.min(), Y.max()],
        origin='lower', cmap='viridis', aspect='auto'
    )
    ax1.set_xlim(x_range)  # Explicitly set the X-axis limits
    ax1.set_ylim((0, 1.0))
    ax1.set_ylabel("y")  # Increase font size for y-label
    ax1.set_title("Magnitude da Velocidade - Re = {} - Δh = {:.3f}".format(RE, dH), fontsize=12)  # Increase font size for title
    ax1.set_aspect('auto')  # Allow the aspect ratio to adjust

    # Add colorbar (legend) on top
    cbar = fig.colorbar(img, ax=ax1, shrink=0.8, aspect=20, orientation='horizontal', location='top', pad=0.15)
    cbar.set_label('Magnitude')  # Increase font size for colorbar label


    # Second subplot: streamlines only
    ax2 = fig.add_axes([0.1, 0.05, 0.8, 0.4])  # Adjusted bottom value from 0.1 to 0.05
    seed_x = np.linspace(x[1], x[-2], 10)  # Adjust number of seeds as needed
    seed_y = np.linspace(y[1], y[-2], 10)

    mask = np.zeros_like(field[0], dtype=bool)
    mask[0:0, 0:0] = True  # Define the region to mask

    masked_field1 = np.ma.array(field[0], mask=mask)
    masked_field2 = np.ma.array(field[1], mask=mask)

    # Apply the X-axis limits to the streamplot
    ax2.streamplot(X[:, :jFinal], Y[:, :jFinal], masked_field1[:, :jFinal], masked_field2[:, :jFinal], color="black", linewidth=0.2, density=4.0, arrowsize=.5)
    ax2.set_xlim(0,x_max)  # Explicitly set the X-axis limits
    ax2.set_ylim((0, eF))
    ax2.set_xlabel("x")  # Increase font size for x-label
    ax2.set_ylabel("y")  # Increase font size for y-label
    ax2.set_title("Linhas de Fluxo".format(name, int(it)))  # Increase font size for title
    ax2.set_aspect('auto')  # Allow the aspect ratio to adjust

    # Set major and minor ticks
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # Major ticks with integer values
    ax2.xaxis.set_minor_locator(AutoMinorLocator(10))  # Minor ticks: you can adjust the number of subticks

    # Show minor ticks
    ax2.tick_params(which='both', axis='x', direction='in', length=6, labelsize=14)  # Increase font size for ticks
    ax2.tick_params(which='minor', axis='x', length=3)
    plt.subplots_adjust(hspace=1.5)  # Adjust the vertical space between subplots

    # Save the figure
    plt.savefig('Frames/VectorFrames/' + "VectorFrame" + "_" + str(int(it)) + '.png')

    # Close the figure to free memory
    plt.close(fig)


def plotPressureField(field, it, name):
    x = np.linspace((e0x - dH / 2), (eFx + dH / 2), Nx + 1)
    x = x / H
    y = np.linspace((e0 - dH / 2), (eF + dH / 2), N + 1)
    X, Y = np.meshgrid(x, y)

    jFinal = (Nx//2)

    # Create the figure and define two subplots
    fig = plt.figure(figsize=(3*X_SIZE, 3*Y_SIZE))  # Adjust figure size to accommodate both plots

    # Main plot (Pressure Field)
    ax_main = fig.add_subplot()
    mask = np.zeros_like(field, dtype=bool)

    mask[0:STEP_I, 0:STEP_J] = True  # Define the region to mask
    masked_field = np.ma.array(field, mask=mask)

    surf = ax_main.pcolormesh(X[:,:jFinal], Y[:,:jFinal], masked_field[:,:jFinal], cmap='viridis')#(0.5*(U0 ** 2)), cmap='viridis')

    # Add color bar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Add color bar next to the plot
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label('Pressure')

    # Labels and title
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.set_title("Pressure Field - Re = {} - Δh = {:.3f}".format(RE,dH))
    ax_main.set_aspect('equal')  # Match aspect ratio


    # Save the figure
    plt.savefig(f'Frames/ScalarFrames/Pressure/{name}_{int(it)}.png')

    # Close
    plt.close()


def plotDivField(field, it, name):
    x = np.linspace((e0x - dH / 2), (eFx + dH / 2), Nx - 1)
    x = x / H
    y = np.linspace((e0 - dH / 2), (eF + dH / 2), N - 1)
    X, Y = np.meshgrid(x, y)

    # Create the figure and define two subplots
    fig = plt.figure(figsize=(3*X_SIZE, 3*Y_SIZE))  # Adjust figure size to accommodate both plots
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)  # GridSpec for layout

    # Main plot (Pressure Field)
    ax_main = fig.add_subplot(gs[0])

    surf = ax_main.pcolormesh(X, Y,field, cmap='viridis')#(0.5*(U0 ** 2)), cmap='viridis')

    # Add color bar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Add color bar next to the plot
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label('Pressure')

    # Labels and title
    ax_main.set_xlabel('x/H')
    ax_main.set_ylabel('y')
    ax_main.set_title("Divergency")
    ax_main.set_aspect('equal')  # Match aspect ratio


    # Save the figure
    plt.savefig(f'Frames/ScalarFrames/Div/{name}_{int(it)}.png')

    # Close
    plt.close()

def plotUScalarField(field, it, name):
    x = np.linspace(e0x, eFx, Nx) #the line intervals
    y = np.linspace(e0-(dH/2), (eF+dH/2),N+1)
    X, Y = np.meshgrid(x  , y) #the complete grid
    X, Y = np.meshgrid(x, y)

    # Create the figure and define two subplots
    fig = plt.figure(figsize=(3*X_SIZE, 3*Y_SIZE))  # Adjust figure size to accommodate both plots
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
    surf = ax_main.pcolormesh(X, Y, (masked_field))

    # Add color bar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Add color bar next to the plot
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label('Velocity')
    ax_main.set_aspect('equal')

    # Labels and title
    ax_main.set_xlabel('x/H')
    ax_main.set_ylabel('y')
    ax_main.set_title("U Velocity Field")

    # Save the figure
    plt.savefig(f'Frames/ScalarFrames/U/{name}_{int(it)}.png')

    # Close
    plt.close()

def plotVScalarField(field, it, name):
    x = np.linspace(e0x-(dH/2), (eFx+dH/2),Nx+1) 
    y =  np.linspace(e0, eF, N) 
    X ,Y = np.meshgrid(x,y)


    # Create the figure and define two subplots
    fig = plt.figure(figsize=(3*X_SIZE, 3*Y_SIZE))  # Adjust figure size to accommodate both plots
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
    surf = ax_main.pcolormesh(X, Y, (masked_field))

    # Add color bar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Add color bar next to the plot
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label('Velocity')
    ax_main.set_aspect('equal')

    # Labels and title
    ax_main.set_xlabel('x/H')
    ax_main.set_ylabel('y')
    ax_main.set_title("V Velocity Field")

    # Save the figure
    plt.savefig(f'Frames/ScalarFrames/V/{name}_{int(it)}.png')

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

def TDMA(a, b, c, d):
    n = len(d)
    w = np.zeros(n-1, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    p = np.zeros(n, dtype=np.float64)
    
    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n-1):
        w[i] = c[i] / (b[i] - a[i-1] * w[i-1])
    
    for i in range(1, n):
        g[i] = (d[i] - a[i-1] * g[i-1]) / (b[i] - a[i-1] * w[i-1])
    
    p[n-1] = g[n-1]
    
    # Reverse loop using forward iteration
    for i in range(n-2, -1, -1):
        p[i] = g[i] - w[i] * p[i+1]
    
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
    div = np.zeros((N+1,Nx+1))
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






# ---------------> SMALL NOTATION COMMENT <---------------
#we use u and v for the vector components, these are in the cells edges
#we use U for the whole vector, this is in the cell center, interpolated back from the edges
#we use x,y for the axis, or X,Y for the whole numpy grid
#our y increases with i




#solving the staggered grid in their points, check notes, this is VERY delicate work, ---> (CHECK THE DIAGRAM, WHICH SHOULD BE IN THE COLAB ATTACHMENT) <----
#note that, in u, there is a extra element in the y axis, both downards and upwards, because of the grid aligment

#in U, our grid is descolated half of dh downards (or upwards, depends in your index definitions) in relation to cell centers
#discrete U grid







#=========Solving ADI, assuming i = Y and j = X, Y growing donwards and X growing sideways, normally===========
#Initial conditions



def plotBoundCondition(upar, yu):
    plt.figure(figsize=(6, 18))  # Adjust the figure size to create the desired aspect ratio
    plt.plot([u * 20 for u in upar], yu)  # Multiply `upar` by 30 for scaling
    plt.xlim(-5, 30)

    plt.title("U Boundary Condition Entry")
    plt.xlabel("u component")
    plt.ylabel("y")

    # Make the aspect ratio 1:3


    plt.show()







@ray.remote
def ADI_U_MID_X(uant_f,vant_f,umid_f):
    bu_f = np.zeros((2,N+1,Nx))
    bu_f[0] = uant_f#exact(Xu,Yu,time)[0]
    bu_f[1] = interpolateVtoU(vant_f)#exact(Xv,Yv,time)[1])
    umid = umid_f.copy()

    for i in range(1,N):
        ro2 = np.zeros(Nx) #velocity term
        ro1 = np.zeros(Nx)

        ro2[:] = (bu_f[1][i,:] * dT) /dH
        ro1[:] = (bu_f[0][i,:] * dT) /dH

        #font, we have the same number in x
        font = np.zeros(Nx-2) 
        font[:] = (1-SIG)*uant_f[i,1:-1] + ((SIG/2.0 - ro2[1:-1]/4) * uant_f[i+1,1:-1])+ ((SIG/2.0 + ro2[1:-1]/4) * uant_f[i-1,1:-1]) 
        font[0] +=  umid_f[i,0] *  (ro1[1]/4 + SIG/2)#border
        font[-1] += umid_f[i,-1] *  (-ro1[-2]/4 + SIG/2) #border

        matrix = tridiagonalX(Nx-2,bu_f,i)
        umid[i,1:-1] = TDMA(matrix[0],matrix[1],matrix[2],font)
    return umid
@ray.remote
def ADI_V_MID_X(uant_f,vant_f,vmid_f):
    bv_f = np.zeros((2,N,Nx+1))
    bv_f[1] = vant_f#exact(Xv,Yv,time)[1]
    bv_f[0] = interpolateUtoV(uant_f)#exact(Xu,Yu,time)[0])
    vmid = vmid_f.copy()

    
    for i in range(1,N-1):
        ro2 = np.zeros(Nx+1) #velocity term
        ro1 = np.zeros(Nx+1)

        ro2[:] = (bv_f[1][i,:] * dT) /dH
        ro1[:] = (bv_f[0][i,:] * dT) /dH

        #our font here is bigger because we have one more term
        font = np.zeros(Nx-1)
        font[:] = (1-SIG)*vant_f[i,1:-1] + ((SIG/2.0 - ro2[1:-1]/4) * vant_f[i+1,1:-1])+ ((SIG/2.0 + ro2[1:-1]/4) * vant_f[i-1,1:-1]) 
        font[0] +=  vmid_f[i,0] * (ro1[1]/4 + SIG/2)#border
        font[-1] += vmid_f[i,-1]*(-ro1[-2]/4 + SIG/2) #border

        matrix = tridiagonalX(Nx-1,bv_f,i)
        vmid[i,1:-1] = TDMA(matrix[0],matrix[1],matrix[2],font)
    return vmid


@ray.remote
def ADI_U_FINAL_Y(umid_f,vmid_f,u_f):
       #for the u component, in the y axis, we have the usual size
    bu_f = np.zeros((2,N+1,Nx))
    bu_f[0] = umid_f#exact(Xu,Yu,time)[0]
    bu_f[1] = interpolateVtoU(vmid_f)#exact(Xv,Yv,time)[1])
    u = u_f.copy()
    for j in range(1,Nx-1):
        ro2 = np.zeros(N+1) #velocity term
        ro1 = np.zeros(N+1)
        ro1[:] = (bu_f[0][:, j] * dT)/dH
        ro2[:] = (bu_f[1][:, j] * dT)/dH

        #we have now a extra term in the font
        font = np.zeros(N-1)
        font[:] = (1-SIG)* umid_f[1:-1,j] + ((SIG/2.0 - ro1[1:-1]/4)* umid_f[1:-1,j+1]) + ((SIG/2.0 + ro1[1:-1]/4)* umid_f[1:-1,j-1]) 
        font[0] += u_f[0,j]* (ro2[1]/4 + SIG/2)
        font[-1] += u_f[-1,j]*(-ro2[-2]/4 + SIG/2) 

        matrix = tridiagonalY(N-1,bu_f,j)
        u[1:-1,j] = TDMA(matrix[0],matrix[1],matrix[2],font)

    return u

@ray.remote
def ADI_V_FINAL_Y(umid_f,vmid_f,v_f):
    bv_f =  np.zeros((2,N,Nx+1))
    bv_f[1] = vmid_f#exact(Xv,Yv,time)[1]
    bv_f[0] = interpolateUtoV(umid_f)#exact(Xu,Yu,time)[0])
    v = v_f.copy()
    
    for j in range(1,Nx):
        ro2 = np.zeros(N) 
        ro1 = np.zeros(N)
        ro1[:] = (bv_f[0][:, j] * dT)/dH
        ro2[:] = (bv_f[1][:, j] * dT)/dH


        font = np.zeros(N-2)
        font[:] = (1-SIG)* vmid_f[1:-1,j] + ((SIG/2.0 - ro1[1:-1]/4)* vmid_f[1:-1,j+1]) + ((SIG/2.0 + ro1[1:-1]/4)* vmid_f[1:-1,j-1]) 
        font[0] += v_f[0,j]* (ro2[1]/4 + SIG/2)
        font[-1] += v_f[-1,j]*(-ro2[-2]/4 + SIG/2) 

        matrix = tridiagonalY(N-2,bv_f,j)
        v[1:-1,j] = TDMA(matrix[0],matrix[1],matrix[2],font)
    return v



def main():

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
    plotVectorField(U,0,"Interpolated",0,10)

    #plotPressureField(P,0,"Pressure")
    #plotPressureField(Pant,0,"Pressure_Exact")


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

    itTime = []


    stable = False
    while stable == False:



        start_time = perf_counter()

        v = stepBoundaryV(v,xv,yv)
        vmid = stepBoundaryV(vmid,xv,yv)
        u = stepBoundaryU(u,xu,yu)
        umid = stepBoundaryU(umid,xu,yu)

        umid_future = ADI_U_MID_X.remote( uant, vant, umid)
        vmid_future = ADI_V_MID_X.remote( uant, vant, vmid)
        umid = ray.get(umid_future)
        vmid = ray.get(vmid_future)

        #with ProcessPoolExecutor() as executor:
        #    umid_future = executor.submit(ADI_U_MID_X, uant, vant, umid)
        #    vmid_future = executor.submit(ADI_V_MID_X, uant, vant, vmid)
        #    umid = umid_future.result()
        #    vmid = vmid_future.result()




        #finished first half step, so we can now increase our time, and repeat the same process
        time = time+dT/2


        v = stepBoundaryV(v,xv,yv)
        vmid = stepBoundaryV(vmid,xv,yv)
        u = stepBoundaryU(u,xu,yu)
        umid = stepBoundaryU(umid,xu,yu)




         # Protect the entry point when running on Windows
        
        u_future = ADI_U_FINAL_Y.remote(umid, vmid, u)
        v_future = ADI_V_FINAL_Y.remote(umid, vmid, v)
        u = ray.get(u_future)
        v = ray.get(v_future)


        #the algorithm has finished the step, so wwe increment the time and swap everything
        #also, we plot the solution, but we have to interpolate it back first
        time+= dT/2
        v = stepBoundaryV(v,xv,yv)
        u = stepBoundaryU(u,xu,yu)
        #solving the pressure now that we have a non-conservative velocity field
        #the procesure is based in the helmholtz decomposition and 
        #the relation between the pressure and velocity that arises when
        #we take the divergence of both sides in the
        #momentum equations - > we have -> laplacian P =  div u
        #whe should be able to solve for P (maybe we can even solve the for the gradient directly)




        #solving P,using the linalg solver for now, we have a extra cell at every edge, because we need the grad, ask questions
        #solve p
        ##pressure solving, final half step
        end_time = perf_counter()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        with open("Error/time_parallel_n={}.csv".format(N), 'a') as file:
            file.write(f'{IT},{execution_time}\n')

        divAntes.append(np.sum(np.abs(calculateDivergency(u,v))))
        B = calculateDivergency(u,v)

        P = np.zeros((N+1,Nx+1))
        Pant = np.zeros((N+1,Nx+1))

        P = stepBoundaryPressure(P,xp,yp,u,v,uant,vant)
        Pant = stepBoundaryPressure(Pant,xp,yp,u,v,uant,vant)


        itCount = 0
        while(True):
            Pant = stepBoundaryPressure(Pant,xp,yp,u,v,uant,vant)
            itCount +=1
            for i in range(1, N):
                for j in range(1, Nx):
                    P[i, j] =  (Pant[i-1, j]+Pant[i+1, j]+Pant[i, j-1] + Pant[i, j+1] + (B[i-1,j-1]*((dH**2)/(dT))))/(4)

            #update fields
            if(np.max(np.abs(P - Pant)) < 1e-12): #se a dif entre ambos for menor que -12, parar
                break
            Pant = P 



        #P = exactPressure(Xp,Yp,time-dT/2)
        P = stepBoundaryPressure(P,xp,yp,u,v,uant,vant)
        Pant = stepBoundaryPressure(Pant,xp,yp,u,v,uant,vant)
        gradPu = calculateUPressureGradient(P)
        gradPv = calculateVPressureGradient(P)

        ##subtract gradP from solution 
        u[1:-1,1:-1] = u[1:-1,1:-1] - (dT)*gradPu[1:-1,1:-1]
        v[1:-1,1:-1] = v[1:-1,1:-1] - (dT)*gradPv[1:-1,1:-1]

    
        v = stepBoundaryV(v,xv,yv)
        vmid = stepBoundaryV(vmid,xv,yv)
        u = stepBoundaryU(u,xu,yu)
        umid = stepBoundaryU(umid,xu,yu)
        P = stepBoundaryPressure(P,xp,yp,u,v,uant,vant)
        Pant = stepBoundaryPressure(Pant,xp,yp,u,v,uant,vant)



        Uant = interpolate(uant,vant)
        U = interpolate(u,v)
        diff = np.abs(U[:,:Nx//2] - Uant[:,:Nx//2])
        print("Pressure converged in {} iterations --- Dif {} -- It = {} -- t = {}".format(itCount,np.max(diff),IT,time),end="\r")
        if(np.max(diff) < 1e-5 and IT > 10): 
            j =  int(((3.06*STEP_HEIGHT) / dH) + 1 + STEP_WIDTH)
            yp = np.linspace((e0 - dH/2),(eF + dH/2),N-1)
            U =interpolate(uant,vant)
            plotVectorField(U,IT,"t = {:.4f}".format(time),x_min=0,x_max=10)
            plotPressureField(Pant,IT,"Pressure_Stable")

            df = pd.DataFrame(interpolate(uant,vant)[0],columns=xp[1:-1],index=yp[:]) #interpolated export of the final field
            df.to_csv("InterpField_Stable_Backstep_re={}.csv".format(RE))


            df = pd.DataFrame(uant,columns=xu[:],index=yu[:]) #interpolated export of the final field
            df.to_csv("UField_Stable_Backstep_re={}.csv".format(RE))
            print("SIMULATION STABLE -- ENDING --  STEP J = {}\n".format(STEP_J))
            stable = True

        uant = u
        vant = v


        umid = np.zeros((N+1,Nx))
        vmid = np.zeros((N,Nx+1))
        u = np.zeros((N+1,Nx))
        v = np.zeros((N,Nx+1))

        Pant = P
        P = np.zeros((N+1,Nx+1))


        #error calculation, we calculate it before the interpolation, in each component


        div.append(np.sum(np.abs(calculateDivergency(uant,vant)))) #talvez dividir por N
        with open("Error/div.csv", 'a') as file:
            file.write(f'{IT},{(np.sum(calculateDivergency(uant,vant)[1:-1,2:-20]))}\n')
        #sample size





        IT +=1
        #
        #with open("Error/errorGrad.csv", 'a') as file:
        #    file.write(f'{N},{max(gradError)}\n')
        # 
        #with open("Error/errorPressure.csv", 'a') as file: 
    #    file.write(f'{N},{max(pError)}\n')

    #plot the divergency graph

if __name__ == '__main__':
    main()