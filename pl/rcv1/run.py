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

# simulate distributed environment with MPI 
comm = MPI.COMM_WORLD
i = comm.Get_rank()
m = max(comm.Get_size(),1)
kappa = 0.9

# Before running the codes, you should download the dataset from "https://www.csie.ntu.edu.tw/~cjlin/libsvm/"
source = 'train.txt'
data = load_svmlight_file(source)

## We define the global varaiables X,y, N,d, i, m,n, A,b, i
X = sp.csr_matrix(data[0]).todense()
y = np.array(data[1])
N,d = X.shape 

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
    pre = 1 / (1 + np.exp(-pre))
    pre1 = np.log(pre)
    pre2 = np.log(1 - pre)
    loss = np.dot(1 + b, pre1.T) + np.dot(1 - b, pre2.T)
    res = -loss / (2 * N) 
    return  res[0,0]

# gradient on the ith agent
def gradient(x, idx=np.arange(n)):
    pre = A[idx,:].dot(x[:-1]) + x[-1]
    pre = 1 / (1 + np.exp(-pre))
    dif = pre - (1 + b[:, idx]) / 2
    dif = dif / N

    grad = AT[:, idx].dot(dif.T)
    grad = np.array(grad)

    grad1 = np.zeros_like(x)
    grad1[:-1] = grad[:,0]
    grad1[-1] = np.sum(dif)

    res = grad1 * n  / len(idx)

    return res

# full gradient for all agents
def full_gradient(x, idx=np.arange(n)):
    pre = X[idx,:].dot(x[:-1]) + x[-1]
    pre = 1 / (1 + np.exp(-pre))
    dif = pre - (1 + y[:, idx]) / 2
    dif = dif / N

    grad = X.T[:, idx].dot(dif.T)
    grad = np.array(grad)

    grad1 = np.zeros_like(x)
    grad1[:-1] = grad[:,0]
    grad1[-1] = np.sum(dif)

    res = grad1 * n  / len(idx)
    return res

# specify the parameters
x0 = np.ones(d + 1)
epochs = 500
print_freq = 2
K = 5

out_fname_DEAREST = './result_DEAREST.csv'
out_fname_DSGD = './result_DSGD.csv'
out_fname_DRONE = './result_DRONE.csv'

sys.path.append("../../optimizers")
from DRONE import DRONE
from DSGD import DSGD 
from DEAREST import DEAREST

B = 256
p = 0.1

if __name__ == "__main__":
    DEAREST(int(1.5*epochs), comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=10, B=B, p=p, fname=out_fname_DEAREST)
    DRONE(epochs, comm,  x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=10, B=16, p=0.1, fname=out_fname_DRONE) 
    DSGD(epochs, comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=10, B=2*B, fname=out_fname_DSGD)


