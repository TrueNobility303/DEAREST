import math
import numpy as np 
import os 
from mixing import fast_mix
from mpi4py import MPI
from tqdm import tqdm
from utils import quantile 

def DRONE(num_iters, comm,  x0, i, m, n, kappa, K, print_freq, fi, gradient, full_gradient, lr, B, p, fname):
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
                                                                        com=comm_cnt,  t=time_cnt, loss=gap, gnorm=gnorm, bnorm=gnorm, mnorm=gnorm), file=file)
    
    ## All the agnets share the same random seed
    np.random.seed(2024)
    
    for epoch in tqdm(range(num_iters)):
        t_begin = MPI.Wtime()

        # compute g
        if epoch == 0:
            x = fast_mix(x, comm, i, m, kappa, K)
            g = gradient(x, np.arange(n))
            s = np.copy(g)
            grad_cnt += n * m
            comp_cnt += n
        else:
            g_old = np.copy(g)
            rad1 = np.random.random()
            if rad1 < p:
                g = gradient(x, np.arange(n))
                grad_cnt += m*n
                comp_cnt += n
            else:
                rad2 = np.random.multinomial(B, [1 / m] * m)
                g_old = np.copy(g)
                if rad2[i]:
                    g += (gradient(x, np.arange(n)) - gradient(x_old, np.arange(n))) * rad2[i] * m / B

                grad_cnt += 2 * np.count_nonzero(rad2) * n 
                comp_cnt += 2 * n 

            # gradient tracking
            s += g - g_old
            s = fast_mix(s, comm, i, m, kappa, K)
            comm_cnt += K

        # update x
        x_old = np.copy(x)
        x -= lr * s
        x = fast_mix(x, comm, i, m, kappa, K)
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

        comm.barrier()
    return