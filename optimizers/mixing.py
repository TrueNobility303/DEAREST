import numpy as np

## mixing on a ring graph, comm shoud be a MPI.COMM_WORLD obejctive

## Codes adapted from "https://github.com/TrueNobility303/DREAM/blob/main/utils/problem.py"

# consensus step in a ring graph
def consensus(variable, comm, rank, size, kappa):
    left = (rank + size - 1) % size
    right = (rank + 1) % size
    send_buffer = np.copy(variable)
    recv_left = np.zeros_like(send_buffer, dtype=float)
    recv_right = np.zeros_like(send_buffer, dtype=float)
    req_left = comm.Isend(send_buffer, dest=left, tag=1)
    req_right = comm.Isend(send_buffer, dest=right, tag=2)
    comm.Recv(recv_left, source=left, tag=2)
    comm.Recv(recv_right, source=right, tag=1)
    req_left.wait()
    req_right.wait()
    return (recv_left * (1-kappa)/2  + recv_right * (1-kappa)/2  + variable * kappa) 


def multi_consensus(variable, comm, rank, size, kappa, K):
    for j in range(K):
        variable =  consensus(variable,comm, rank, size,kappa)
    return variable

def fast_mix(variable, comm, rank, size, kappa, K):
    variable_old = np.copy(variable)
    # calculate the second largest singular value
    sig = kappa + (1-kappa) * np.cos(2 * (size -1) / size * np.pi)
    q = np.sqrt(1-sig*sig)
    eta = (1-q) / (1+q)  
    for j in range(K):
        variable = (1+eta) * consensus(variable,comm,rank, size,kappa) - eta * variable_old
        variable_old = np.copy(variable)
    return variable