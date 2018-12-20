import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

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

def plot_speedup(procs, times, ser_time, label=None):
    speedup = ser_time / times
    plt.plot(procs, speedup, label=label)

# Get run data
with open('out.txt') as f:
    content = f.readlines()
content = [float(x.strip()) for x in content]
procs = np.array(content[0::4])
times = np.array(content[1::4])
rands = np.array(content[2::4])
ser_time = rands[0]
times[0] = ser_time

# Get plot title
title = ''
if (cl_args.n_nodes is not None and cl_args.edge_prob is not None):
    title = 'Random Graph: N=' + str(cl_args.n_nodes) + \
    ', P=' + str(cl_args.edge_prob) + \
    ', lr=' + str(cl_args.lr) + \
    ', seed=' + str(cl_args.seed) + \
    ', iter=' + str(cl_args.n_iter)
elif (cl_args.L is not None and cl_args.w is not None):
    title = 'Par Graph: L=' + str(cl_args.L) + \
    ', w=' + str(cl_args.w) + \
    ', lr=' + str(cl_args.lr) + \
    ', seed=' + str(cl_args.seed) + \
    ', iter=' + str(cl_args.n_iter)
else:
    title = 'User Graph: N=' + str(len(cl_args.n_nodes)) + \
    ', Iter=' + str(cl_args.n_iter) + \
    ', lr=' + str(cl_args.lr) + \
    ', seed=' + str(cl_args.seed) + \
    ', iter=' + str(cl_args.n_iter)

# Plot speedup
linear = procs
plt.plot(procs, linear, label='Perfect Linear')
plot_speedup(procs, times, ser_time, label='Observed Speedup')
plot_speedup(procs, rands, ser_time, label='Random Assignment Speedup')
plt.title(title)
plt.xlabel('# of Processors')
plt.ylabel('Speedup')
plt.legend(loc='best')
plt.savefig('speedup.png')