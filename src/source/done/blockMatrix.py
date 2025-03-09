import numpy as np

from time import  perf_counter_ns
from scipy.sparse import diags
import sys

t1 = 0

def BTDMA(A, b, N):
    """
    Solves the block tridiagonal system Ax = b, where A is composed of N by N blocks.

    Parameters:
    A : ndarray
        Block tridiagonal matrix of shape (rows, cols) where rows = cols = N * number of blocks.
    b : ndarray
        Right-hand side vector of shape (rows,).
    N : int
        Size of the block.

    Returns:
    x : ndarray
        Solution vector.
    """
    rows, cols = A.shape

    if rows % N != 0:
        raise ValueError("A must have an integer number of N by N blocks")

    # Get number of blocks
    nblk = rows // N


    b = b.reshape(nblk, N)  
    #print(b)



    x = np.zeros((nblk, N))
    c = np.zeros((nblk, N))

    D = np.zeros((nblk,N, N))
    Q = np.zeros((nblk,N, N))
    G = np.zeros((nblk-1,N, N))
    C = np.zeros((nblk-1,N, N))
    B = np.zeros((nblk-1,N, N))

    # Convert A into arrays of blocks
    for k in range(nblk - 1):
        block = slice(k * N, (k + 1) * N)

        D[k,:, :] = A[block, block]
        B[k,:, :] = A[(k + 1) * N:(k + 2) * N, block]
        C[k,:, :] = A[block, (k + 1) * N:(k + 2) * N]

    #start from here
    global t1
    t1 = perf_counter_ns()


    # Last diagonal block
    D[ nblk - 1,:, :] = A[(nblk - 1) * N:nblk * N, (nblk - 1) * N:nblk * N]

    # Forward sweep
    Q[0,:, :] = D[0,:, :]

    G[0,:, :] = np.linalg.solve(Q[0,:, :], C[0,:, :])

    for k in range(1, nblk - 1):
        Q[k,:, :] = D[k,:, :] - B[k-1,:, :] @ G[k-1,:, :]
        G[k,:, :] = np.linalg.solve(Q[k,:, :], C[k,:, :])
    Q[nblk - 1,:, :] = D[ nblk - 1,:, :] - B[nblk - 2,:, :] @ G[nblk - 2,:, :]

    c[ 0,:] = np.linalg.solve(Q[0,:, :], b[0, :]) 
    for k in range(1, nblk):
        c[k,:] = np.linalg.solve(Q[k,:, :], b[k,:] - B[k - 1,:, :] @ c[k - 1,:])

    # Back substitution
    x[nblk - 1,:] = c[nblk - 1,:]
    for k in range(nblk - 2, -1, -1):
        x[ k,:] = c[ k,:] - G[k,:, :] @ x[k + 1,:]

    # Revert to vector form
    return x.flatten()

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

blockSize = 2
SIZE = int(sys.argv[1])


#TDMA Matrix
mainDiag = np.ones(SIZE)
lowerDiag = np.ones(SIZE-1) * 0.5
upperDiag = np.ones(SIZE-1) * 0.5
diagonals = [mainDiag, upperDiag, lowerDiag]
positions = [0, 1, -1] 

MAT = diags(diagonals, positions, shape=(SIZE,SIZE)).toarray()
FONTE = np.ones(SIZE)* 2.5
#print("MAT TDMA")
#print(MAT)
t1n = perf_counter_ns()

x = TDMA(lowerDiag,mainDiag,upperDiag,FONTE)
x = TDMA(lowerDiag,mainDiag,upperDiag,FONTE)

t2n = perf_counter_ns()
print("Time TDMA {}".format(t2n-t1n))


#BTDMA has twice the size
SIZE = SIZE*2
mainDiag = np.ones(SIZE)
lowerDiag = np.ones(SIZE-1) * 0.5
upperDiag = np.ones(SIZE-1) * 0.5
diagonals = [mainDiag, upperDiag, lowerDiag]
positions = [0, 2, -2]  

MAT = diags(diagonals, positions, shape=(SIZE,SIZE)).toarray()
FONTE = np.ones(SIZE)* 2.5
#print("MAT BTDMA")
#print(MAT)




x = BTDMA(MAT,FONTE,blockSize)

t2 = perf_counter_ns()
print("Time BTDMA {}".format(t2-t1))
print("Ratio {}".format((t2-t1)/(t2n-t1n)))


with open("Error/ExptimeTDMA.csv", 'a') as file:
    file.write(f'{SIZE/2},{t2n-t1n}\n')
with open("Error/ExptimeBTDMA.csv", 'a') as file:
    file.write(f'{SIZE/2},{t2-t1}\n')










