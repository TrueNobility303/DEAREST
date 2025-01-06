import math
import numpy as np 
import os 
from mixing import multi_consensus
from mpi4py import MPI
from tqdm import tqdm
from utils import quantile 

# DSGD from the implementation of https://github.com/liboyue/Network-Distributed-Algorithm/blob/master/nda/optimizers/decentralized_distributed/DSGD.py
def DSGD(num_iters, comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr, B,  fname):
    B_per_agent =  math.ceil(B / m)
    if os.path.exists(fname) and i == 0:
        return
    # store num_grad, num_com, time and f(x_out) every print_freq iterations
    grad_cnt = 0
    comm_cnt = 0
    time_cnt = 0
    comp_cnt = 0

    # initialize
    x = np.copy(x0)
    xx = np.zeros_like(x)
    bx = np.zeros_like(x)
    g = np.zeros_like(x)
    s = np.zeros_like(x)
    fix = np.array([fi(x), 1.0])
    loss = np.array([0.0, 1.0])

    # record
    comm.Reduce(fix, loss, op=MPI.SUM)
    if i == 0:
        gap = loss[0] / n
        gnorm = np.linalg.norm(full_gradient(x)) 
        with open(fname, 'w') as file:
            print('{grad:d},{comp:d}, {com:d}, {t:.3f},{loss:.15f}, {gnorm:.15f}, {bnorm:.15f}, {mnorm:.15f}'.format(grad=grad_cnt, comp=comp_cnt,
                                                                        com=comm_cnt,  t=time_cnt, loss=gap,  gnorm=gnorm, bnorm=gnorm, mnorm=gnorm), file=file)

    np.random.seed(2024)
    for epoch in tqdm(range(num_iters)):
        t_begin = MPI.Wtime()

        idx = np.random.choice(range(n), B_per_agent)
        
        # compute g
        if epoch == 0:
            g = gradient(x, idx)
            s = np.copy(g)
            grad_cnt += B_per_agent * m
            comp_cnt += B_per_agent
        else:
            g_old = np.copy(g)
            g = gradient(x, idx)
            grad_cnt += B_per_agent * m
            comp_cnt += B_per_agent

            # gradient tracking
            s += g - g_old
            s = multi_consensus(s, comm, i, m, kappa, K=1)
            comm_cnt += K

        # update x
        x -= lr / (epoch+1) * s
        x = multi_consensus(x, comm, i, m, kappa, K=1)
        t_end = MPI.Wtime()
        comm_cnt += K
        time_cnt += (t_end - t_begin)

        # record
        if (epoch + 1) % print_freq == 0:
            rad3 = np.random.randint(0, m)
            xx = comm.bcast(x, root=rad3)
            fix[0] = fi(xx)
            comm.barrier()
            comm.Reduce(x, bx, op=MPI.SUM)
            bx /= m
           
            comm.Reduce(fix, loss, op=MPI.SUM)
            agent_gnorm = np.linalg.norm(full_gradient(x))
            gather_gnorm = comm.gather(agent_gnorm, root=0)
            if i == 0:
                gap = loss[0] / m
                bnorm, gnorm, mnorm = quantile(gather_gnorm)
                with open(fname, '+a') as file:
                    print('{grad:d},{comp:d}, {com:d}, {t:.3f},{loss:.15f}, {gnorm:.15f}, {bnorm:.15f}, {mnorm:.15f}'.format(grad=grad_cnt, comp=comp_cnt,
                                                                        com=comm_cnt,  t=time_cnt, loss=gap, gnorm=gnorm, bnorm=bnorm, mnorm=mnorm), file=file)
                    
    return