import numpy as np
from mpi4py import MPI
from tqdm import tqdm
import os
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 16
m = size 
kappa = 0.9 

# specify the parameters(delta=1,mu=1)
# ep = 0.001
# t = 2 * math.floor(math.log(1.5 * ep) / math.log(7 / 8))

T = 100
beta = 1000
t  = 5 
d = T * t * m * n 
x0 = np.zeros(d)

epochs = T * t 
print_freq = 5

b = np.zeros(T * t)
for i in range(T * t):
    j = math.floor(i / T)
    b[i] = (7 ** j) / (8 ** j)

def psi(x, theta):
    if x / theta < 31 / 32:
        return 0.5 * x ** 2
    elif x / theta < 1:
        return 0.5 * x ** 2 - 16 * (x - 31 / 32 * theta) ** 2
    elif x / theta < 33 / 32:
        return 0.5 * x ** 2 - theta ** 2 / 32 + 16 * (x - 33 / 32 * theta) ** 2
    else:
        return 0.5 * x ** 2 - theta ** 2 / 32

def qtt(X):
    ans = 0
    for i in range(t):
        ans += (7 / 8 * (0 if i == 0 else X[i * T - 1]) - X[i * T]) ** 2
        for j in range(T - 1):
            ans += (X[i * T + j + 1] - X[i * T + j]) ** 2
    ans /= 2
    return ans

def fvalue_individual(X):
    X_ = b - beta * X
    ans = qtt(X_)
    for i in range(T*t):
        ans += psi(X_[i], b[i])
    return ans / (beta*beta)

def r_grad(X):
    grad = b - beta * X
    for j in range(T*t):
        grad[j] -= max(0, b[j] - 32 * beta * abs(X[j]))
    grad = -grad / beta
    return grad


def q1_grad(X):
    X_ = b - beta * X
    grad = np.zeros(T*t)
    for j in range(T*t):
        if j // 2 == 0:
            grad[j] = X_[j] - X_[j + 1]
        else:
            grad[j] = -grad[j - 1]
    grad = - grad / beta
    return grad


def q2_grad(X):
    X_ = b - beta * X
    grad = np.zeros(T*t)
    for i in range(t):
        if i == 0:
            grad[0] = X_[0]
        else:
            grad[i * T - 1] = 7 / 8 * (7 / 8 * X_[i * T - 1] - X_[i * T])
            grad[i * T] = X_[i * T] - 7 / 8 * X_[i * T - 1]
        if T > 2:
            for j in range(i * T // 2 + 1, (i + 1) * T // 2):
                grad[2 * j - 1] = X_[2 * j - 1] - X_[2 * j]
                grad[2 * j] = -grad[2 * j - 1]
    grad = -grad / beta 
    return grad

def grad_individual(x):
    g = r_grad(x) + q1_grad(x) + q2_grad(x) 
    return g 

def gradient(x, idx=np.arange(n)):
    g = np.zeros(d)
    i = rank
    for j in idx:
        g[i*j*T*t:i*j*T*t+T*t] = grad_individual(x[i*j*T*t:i*j*T*t+T*t])
    return g / len(idx)

def full_gradient(x):
    g = np.zeros(d)
    for i in range(m):
        for j in range(n):
            g[i*j*T*t:i*j*T*t+T*t] = grad_individual(x[i*j*T*t:i*j*T*t+T*t])
    return g / (m*n)

def fi(x):
    res = 0
    i = rank
    for j in range(n):
        res += fvalue_individual(x[i*j*T*t:i*j*T*t+T*t])
    return res / n 

print_freq = 5
K = 5

out_fname_DRONE = './result_DRONE.csv'
out_fname_DEAREST = './result_DEAREST.csv'
out_fname_DSGD = './result_DSGD.csv'

B_DSGD = 16 
B_DRONE = 16
p_DRONE = 2 * B_DRONE / (2*B_DRONE +n)

B_DEAREST = 16
p_DEAREST = 2 * B_DEAREST / (2*B_DEAREST +m*n)

import sys 
sys.path.append("../../optimizers")
from DRONE import DRONE
from DSGD import DSGD 
from DEAREST import DEAREST

if __name__ == "__main__":
    DEAREST(epochs, comm, x0, rank, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1, B=B_DEAREST, p=p_DEAREST, fname=out_fname_DEAREST)
    DSGD(epochs, comm, x0, rank, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1, B=2*B_DSGD, fname=out_fname_DSGD)
    DRONE(epochs, comm,  x0, rank, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr=1, B=B_DRONE, p=p_DRONE, fname=out_fname_DRONE) 
    
    





