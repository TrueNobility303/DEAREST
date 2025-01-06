import math
import numpy as np 
import os 
from mixing import multi_consensus
from mpi4py import MPI
from tqdm import tqdm
from utils import quantile 

## GT-SARAH from "Fast decentralized non-convex finite-sum optimization with recursive variance reduction" in SIOPT 2020.
def GTSARAH(num_iters, comm, x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr, B, freq, fname):
    B_per_agent = math.ceil(B / m)

    if os.path.exists(fname) and i == 0:
        return
    # store num_grad, num_com, time and f(x_out) every print_freq iterations
    grad_cnt = 0
    comm_cnt = 0
    time_cnt = 0
    comp_cnt = 0
    # initialize
    x = np.copy(x0)
    x_old = np.copy(x)
    xx = np.copy(x)
    g = np.zeros_like(x)
    s = np.zeros_like(x)
    fix = np.array([fi(x), 1.0])
    loss = np.array([0.0, 1.0])
    bx = np.zeros_like(x)

    # record
    comm.Reduce(fix, loss, op=MPI.SUM)
    if i == 0:
        gap = loss[0] / m
        gnorm = np.linalg.norm(full_gradient(x)) 
        with open(fname, 'w') as file:
            print('{grad:d},{comp:d}, {com:d}, {t:.3f},{loss:.15f}, {gnorm:.15f}, {bnorm:.15f}, {mnorm:.15f}'.format(grad=grad_cnt, comp=comp_cnt,
                                                                        com=comm_cnt,  t=time_cnt, loss=gap,  gnorm=gnorm, bnorm=gnorm, mnorm=gnorm), file=file)
    

    for epoch in tqdm(range(num_iters)):
        t_begin = MPI.Wtime()
        if epoch == 0:
            g = gradient(x, np.arange(n))
            s = multi_consensus(g, comm, i, m, kappa, K=1)
            comm_cnt += K
            grad_cnt += m*n
            comp_cnt += n
        else:
            g_old = np.copy(g)
            if epoch % freq == 0:
                g = gradient(x, np.arange(n))
                grad_cnt += m*n
                comp_cnt += n
            else:
                np.random.seed(2024+i)
                idx = np.random.choice(range(n), B_per_agent, replace=True)
                grad_cnt += 2 * B_per_agent * m 
                comp_cnt += 2 * B_per_agent
                g += (gradient(x, idx) - gradient(x_old, idx))

            # gradient tracking
            s += g - g_old
            s = multi_consensus(s, comm, i, m, kappa, K=1)
            comm_cnt += 1

        # update x
        x_old = np.copy(x)
        x -= lr * s
        x = multi_consensus(x, comm, i, m, kappa, K=1)
        t_end = MPI.Wtime()
        comm_cnt += 1
        time_cnt += (t_end - t_begin)
        
        # record
        if (epoch + 1) % print_freq == 0:
            np.random.seed(2024)
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
                # gnorm = np.linalg.norm(gradient(x))
                # bnorm = np.linalg.norm(gradient(bx)) 
                # mnorm = np.max(gather_gnorm)
                bnorm, gnorm, mnorm = quantile(gather_gnorm)
                with open(fname, '+a') as file:
                    print('{grad:d},{comp:d}, {com:d}, {t:.3f},{loss:.15f}, {gnorm:.15f}, {bnorm:.15f}, {mnorm:.15f}'.format(grad=grad_cnt, comp=comp_cnt,
                                                                        com=comm_cnt,  t=time_cnt, loss=gap, gnorm=gnorm, bnorm=bnorm, mnorm=mnorm), file=file)

        comm.barrier()
    return
