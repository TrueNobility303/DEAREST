# mpiexec -n 16 python run.py 
import numpy as np
import pandas as pd
from mpi4py import MPI
from tqdm import tqdm
import os
import math
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file 
import sys 
import scipy 

# simulate distributed environment with MPI 
comm = MPI.COMM_WORLD
i = comm.Get_rank()
m = max(comm.Get_size(),1)
kappa = 0.9

mat = scipy.io.loadmat('wikivital_mathematics.mat')
X = mat['A']
y = mat['b']

N, d = X.shape

## Split the data and drop the last elements 
n = N // m
N = n * m
X = X[:N, :]
y = y[:N]
y = y.reshape((1,len(y)))

## Get the data on the ith machine, denoted by (A,b) 
A = X[i*n:(i+1)*n,:]
AT = A.T
b = y[:, i*n:(i+1)*n]


# loss
def fi(x):
    pre = A.dot(x[:-1]) + x[-1]
    dif = pre - b
    loss = dif.T.dot(dif) 
    res = loss / (2 * N) 
    return  res[0,0]

# gradient on the ith agent
def gradient(x, idx=np.arange(n)):
    pre = A[idx,:].dot(x[:-1]) + x[-1]
    dif = pre - b[:,idx]
    grad = AT[:, idx].dot(dif.T)
    grad = np.array(grad)

    grad1 = np.zeros_like(x)
    grad1[:-1] = grad[:,0]
    grad1[-1] = np.sum(dif)

    res = grad1 / len(idx) 

    return res

# full gradient
def full_gradient(x, idx=np.arange(n)):
    pre = X[idx,:].dot(x[:-1]) + x[-1]
    dif = pre - y[:, idx]
    grad = X.T[:, idx].dot(dif.T)
    grad = np.array(grad)

    grad1 = np.zeros_like(x)
    grad1[:-1] = grad[:,0]
    grad1[-1] = np.sum(dif)

    res = grad1 / len(idx)
    return res

# specify the parameters
x0 = np.zeros(d + 1)
epochs = 500
print_freq = 2
K = 5

out_fname_DRONE = './result_DRONE.csv'
out_fname_DEAREST = './result_DEAREST.csv'
out_fname_DSGD = './result_DSGD.csv'

sys.path.append("../../optimizers")
from DEAREST import DEAREST 
from DSGD import DSGD 
from DRONE import DRONE

B = 64
p = 2*B / (2*B + m*n)
if __name__ == "__main__":
    DSGD(3*epochs, comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1e-4, B=2*B, fname=out_fname_DSGD)
    DRONE(epochs, comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1e-4, B=4, p=1/3, fname=out_fname_DRONE)
    DEAREST(epochs, comm,  x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1e-4, B=B, p=p, fname=out_fname_DEAREST) 
    
    
    




