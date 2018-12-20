from mpi4py import MPI
import numpy as np
import argparse
import time
from utils import fit, window_avg, time_str
from layers import Input, Flatten, Add, Linear, Conv1d, Timing
from controller import Controller
from distGraph import Graph, RandomGraph
from session import Session

# Begin timing
t_start = time.time()

# Get command line args
parser = argparse.ArgumentParser(description='Computation Graph Scheduling with RL')
parser.add_argument('--lr', type=float, default=1e-3, metavar='l',
                    help='Controller learning rate (default=1e-3)')
parser.add_argument('--use_baseline', type=bool, default=True, metavar='B',
                    help='Boolean specifying whether controller uses baseline in loss calculation (default=True)')
parser.add_argument('--n_iter', type=int, default=1000, metavar='I',
                    help='Number of controller iterations (default=1000)')
parser.add_argument('--n_nodes', type=int, default=None, metavar='N',
                    help='Number of graph nodes if using random graph (default=None)')
parser.add_argument('--edge_prob', type=float, default=None, metavar='P',
                    help='Probability that an edge exists between any two nodes if using random graph (default=None)')
parser.add_argument('--L', type=int, default=None, metavar='L',
                    help='Number of layers in ParGraph (default=None)')
parser.add_argument('--w', type=int, default=None, metavar='w',
                    help='Layer width in ParGraph (default=None)')
parser.add_argument('--timing_runs', type=int, default=1, metavar='T',
                    help='Number of runs of the computation graph to average over for timing (default=10)')
parser.add_argument('--seed', type=int, default=None, metavar='s',
                    help='Numpy random seed (default=None)')
parser.add_argument('--verbose', type=int, default=0, metavar='v',
                    help='Set verbosity level (default=0, only final output)')
cl_args = parser.parse_args()

# Init MPI
comm = MPI.COMM_WORLD

# Fix random seed across processes
if cl_args.seed is None:
    seed = None
    if comm.rank == 0:
        seed = np.random.randint(2**32)
    seed = comm.bcast(seed, root=0)
    np.random.seed(seed)
    cl_args.seed = seed
else:
    np.random.seed(cl_args.seed)

# Create Session instance
sess = Session(comm, cl_args)

# Initialize the graph
t=0.001
input_shape = (3000,1)
if (cl_args.n_nodes is not None and cl_args.edge_prob is not None):
    sess.init_random_graph(t=t, input_shape=input_shape)
elif (cl_args.L is not None and cl_args.w is not None):
    sess.init_par_graph(t=t, input_shape=input_shape)
else: # User defined graph
    node_list = [
        Input(),                                #0
        Timing(input_shape=input_shape, t=t),   #1
        Timing(input_shape=input_shape, t=t),   #2
        Timing(input_shape=input_shape, t=t),   #3
        Timing(input_shape=input_shape, t=t),   #4
        Timing(input_shape=input_shape, t=t),   #5
        Timing(input_shape=input_shape, t=t),   #6
        Timing(input_shape=input_shape, t=t),   #7
        Timing(input_shape=input_shape, t=t),   #8
        Timing(input_shape=input_shape, t=t),   #9
        Timing(input_shape=input_shape, t=t),   #10
        Timing(input_shape=input_shape, t=t),   #11
        Timing(input_shape=input_shape, t=t),   #12
        Add()                                   #13
    ]
    input_idx_list = [0, 0, 0, 0, 2, 3, 1, 4, 8, 7, 5, 6, [9,10,11,12]]
    sess.init_graph(node_list, input_idx_list, input_shape)

# Learn the assignments
sess.learn_assignments()

# Report total runtime
t_end = time.time()
if comm.rank == 0:
    if cl_args.verbose > 0:
        print("Total Runtime:              " + time_str(t_end-t_start))
    else:
        print(t_end-t_start)