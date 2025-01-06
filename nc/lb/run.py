# mpiexec -n 16 python run.py 

import numpy as np
import pandas as pd
from mpi4py import MPI
from tqdm import tqdm
import os
import math
import scipy.sparse as sp
from scipy.stats import norm

# simulate distributed environment with MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
m = max(comm.Get_size(), 1)
kappa = 0.9

# problem setup
n = 16
sigma = 1e-3

# problem dimension 
T = 100
d = m*n*T
epochs = 100

## the worse case function by Carmon Y, Duchi JC, Hinder O, Sidford A. "Lower bounds for finding stationary points I. Mathematical Programming. 2020"
def Psi(x):
    if x < 0.5:
        return 0
    else:
        return math.exp(1 - 1 / (2*x-1)**2) 

def grad_Psi(x):
    if x < 0.5:
        return 0
    else:
        y = 2*x - 1
        return -4 / y**3 * math.exp(1 - 1/y**2)

def Phi(x):
    return math.sqrt( 2 * math.pi * math.e) * norm.cdf(x)

def grad_Phi(x):
    return math.sqrt(math.e) * math.exp( - x**2 /2)

def grad_individual(x):
    g = np.zeros(T)
    x = x / sigma
    for j in range(T):
        if j == 0:
            g[j] = -Psi(-1) * grad_Phi(-x[j]) - Psi(1) * grad_Phi(x[j]) - grad_Psi(-x[j]) * Phi(-x[j+1]) - grad_Psi(x[j]) * Phi(x[j+1])
        elif j == T-1:
            g[j] = -Psi(-x[j-1]) * grad_Phi(-x[j]) - Psi(x[j-1]) * grad_Phi(x[j])
        else:
            g[j] = -Psi(-x[j-1]) * grad_Phi(-x[j]) - Psi(x[j-1]) * grad_Phi(x[j]) - grad_Psi(-x[j]) * Phi(-x[j+1]) - grad_Psi(x[j]) * Phi(x[j+1])
    return sigma * g 

def fvalue_individual(x):
    x = x / sigma 
    res = -Psi(1) * Phi(x[0])
    for j in range(T):
        res += Psi(-x[j-1]) * Phi(-x[j]) - Psi(x[j-1]) * Phi(x[j])
    return sigma * sigma * res
 
def gradient(x, idx=np.arange(n)):
    g = np.zeros(d)
    i = rank
    for j in idx:
        g[i*j*T:i*j*T+T] = grad_individual(x[i*j*T:i*j*T+T])
    return g / len(idx)

def full_gradient(x):
    g = np.zeros(d)
    for i in range(m):
        for j in range(n):
            g[i*j*T:i*j*T+T] = grad_individual(x[i*j*T:i*j*T+T])
    return g / (m*n)

def fi(x):
    res = 0
    i = rank
    for j in range(n):
        res += fvalue_individual(x[i*j*T:i*j*T+T])
    return res / n 

# specify the parameters
x0 = np.zeros(d)
print_freq = 2
K = 5

out_fname_DESTRESS = './result_DESTRESS.csv'
out_fname_DEAREST = './result_DEAREST.csv'
out_fname_DSGD = './result_DSGD.csv'

B = 16
p = 2 * B / (2*B +m*n)

import sys 
sys.path.append("../../optimizers")
from DESTRESS import DESTRESS 
from DSGD import DSGD 
from DEAREST import DEAREST

if __name__ == "__main__":
    DESTRESS(epochs, comm,  x0, rank, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=5e-2, B=B, freq=B, fname=out_fname_DESTRESS)  # for DESTRESS lr=1e-1 will diverge due to its instability
    DSGD(epochs, comm, x0, rank, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1e-1, B=2*B, fname=out_fname_DSGD)
    DEAREST(epochs, comm, x0, rank, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1e-1, B=B, p=p, fname=out_fname_DEAREST)




