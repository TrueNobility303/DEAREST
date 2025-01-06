# mpiexec -n 32 python run.py 
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

## Calculate the smoothness coefficients

# L = 1/4 * np.linalg.norm(X) ** 2 / N
## estimate 2-norm with power method
# def estimate_L(A, num_iters=10):
#     m = A.shape[0]
#     v = np.random.normal(size=(m,1))
#     for j in range(num_iters):
#         v =  A @ (A.T @ v)
#         v = v / np.linalg.norm(v)
#     sigma_max = np.linalg.norm(A.T @ v)
#     return sigma_max 

# L = 1 / 4 *  estimate_L(X) / N
# bar_L = 1 / 4 * np.linalg.norm(X) / N 
    
# max_L = 0
# max_bar_L = 0
# for j in range(m):
#     Aj = X[j*n:(j+1)*n,:]
#     Li = 1 / 4 *  estimate_L(A) / n 
#     bar_Li = 1 / 4 * np.linalg.norm(A) / n 
#     if Li > max_L:
#         max_L = Li 
#     if  bar_Li > max_bar_L:
#         max_bar_L = bar_Li

# if i == 0:
#     print(f'L: {L} | bar L: {bar_L}')
#     print(f'L Local: {max_L} | bar L Local: {max_bar_L}')

# The results on "rcv1" dataset with 32 agents
# L: 0.0002617735639123714 | bar L: 0.0017579498451969122
# L Local: 0.0019007422320240373 | bar L Local: 0.009944466052196626

theta = 1e-8
nu = 10

# loss
def fi(x):
    pre = A.dot(x[:-1]) + x[-1]
    pre = 1 / (1 + np.exp(-pre))
    pre1 = np.log(pre)
    pre2 = np.log(1 - pre)
    loss = np.dot(1 + b, pre1.T) + np.dot(1 - b, pre2.T)
    reg = theta * np.sum(nu * x * x / (nu * x * x + 1))
    res = -loss / (2 * N) +  reg 
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

    denominator = (1 + nu * x * x) * (1 + nu * x * x)
    numerator = 2 * theta * nu * x
    grad2 = numerator / denominator
    res = grad1 * n  / len(idx) + grad2 

    return res

# full gradient
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

    denominator = (1 + nu * x * x) * (1 + nu * x * x)
    numerator = 2 * theta * nu * x
    grad2 = numerator / denominator
    res = grad1 * n  / len(idx) + grad2 
    return res

# specify the parameters
x0 = np.ones(d + 1)
epochs = 500
print_freq = 2
K = 5

out_fname_DESTRESS = './result_DESTRESS.csv'
out_fname_DEAREST = './result_DEAREST.csv'
out_fname_DSGD = './result_DSGD.csv'
out_fname_GTSARAH = './result_GTSARAH.csv'
#out_fname_DESTRESS_plus = './result_DESTRESS_plus.csv'

## The parameters of previous works DESTRESS depend on the local smoothness
B_local = math.ceil(math.sqrt(m*n))

S_local = B_local 
p_local = 2*B_local / (2*B_local + m*n)
# Following the theoretical value of DESTRESS and GT-SARAH

## In our work (DEAREST), we differ the local and global smoothness 
# We use variable tau to estimate bar L / L

tau = 10
sig = kappa + (1-kappa) * np.cos(2 * (m -1) / m * np.pi)

# theoretical step sizes 
# lr_GT_SARAH = (1-sig)*(1-sig)/( 8 * math.sqrt(42) *bar_L)
# lr_DESTRESS = 1/(640*bar_L)
# lr_DEAREST = 1/(8*L)

B_global = tau * math.ceil(math.sqrt(m*n))
p_global = 2*B_global / (2*B_global + m*n)
## Parameter p is chosen such that p*m*n = (1-p)*2*b, which makes each loopness iteration of the same cost 

sys.path.append("../../optimizers")
from DESTRESS import DESTRESS 
from DSGD import DSGD 
from DEAREST import DEAREST
from GTSARAH import GTSARAH 

B = 256
p = 2*B / (2*B + m*n)
if __name__ == "__main__":
    DESTRESS(epochs, comm,  x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1, B=B, freq=B, fname=out_fname_DESTRESS) # for DESTRESS lr=10 will diverge due to its instability
    DSGD(epochs, comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=10, B=2*B, fname=out_fname_DSGD)
    DEAREST(epochs, comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=10, B=B, p=p, fname=out_fname_DEAREST)
    #GTSARAH(tau*epochs,comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=10, B=B_local, freq=S_local, fname=out_fname_GTSARAH)
    
## Log of tunning paramters for DEAREST and DESTRESS, 
## 2024/12/11 22：50 B=64 m=16 lr both be 1 , similar performances
## 2024/12/11 22:56  B=64 m=16 lr both be 10, DESTRESS fails further tuned to be 1 still fails
## 2024/12/11 23:45 try B=256 m=16 lr both be 10 DESTRESS fails further tuned to be 1 (running DESTRESS and DSGD)
    




