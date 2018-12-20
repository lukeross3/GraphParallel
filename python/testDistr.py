# Run with command: 'mpiexec -np 2 python3 python/testDistr.py'

from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # automatic MPI datatype discovery
    data = np.empty(10, dtype=np.float64)
    if rank == 0:
        data = np.arange(10, dtype=np.float64)
        comm.Send(data, dest=1, tag=13)
    elif rank == 1:
        comm.Recv(data, source=0, tag=13)

    print(str(rank) + "/" + str(size) + ": " + str(data))

if __name__ == '__main__':
    main()
