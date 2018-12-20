# GraphParallel
Problem: given a computation graph with N nodes, assign each node to one of p processors to minimize parallel run-time of the computation graph.

Solution: graph search using reinforcement learning.

## Usage
The file `python/testDistr.py` is given to test the MPI backend used to communicate between processors, and the file `python/main.py` runs the learning process on a single computation graph.  A random graph or parallel permutation graph can be specified via the command line.  Otherwise if neither is specified then the user defined graph in `python/main.py` is used.  Run `python3 python/main.py -h` for a description of the different command line arguments.  To run the learning process on "num_procs" processors, run the following command:
```
mpiexec -np <num_procs> python3 python/main.py <cl_args>
```
Sample output:

![LearningCurve](https://github.com/lukeross3/GraphParallelPrivate/blob/public/images/rand_100_learn_10000.png)

Scripts are provided to test and plot the speedup of the method as the number of processors increases (graphs are specified similarly).  Starter PBS scripts are also provided to submit jobs to a PBS job scheduler.  To test speedup locally, use one of the local speedup scripts:
```
./scripts/local_speedup_rand.sh <cl_args>
```
Or:
```
./scripts/local_speedup_par.sh <cl_args>
```
Sample output:

![SpeedupCurve](https://github.com/lukeross3/GraphParallelPrivate/blob/public/images/rand_100_sqrt_5000_local_speedup.png)


The two scripts correspond to two types of computation graphs, discussed in [Parallel Computation Graphs](#parallel-computation-graphs).

## Reinforcement Learning Search
The problem addressed in this work is as follows: given a directed, acyclic graph with N nodes, assign each
node to one of p processors to minimize parallel run-time of the computation graph.  This yields a combinatorial graph search, as there are p^N ways to complete this assignment (for a tighter bound accounting for the fact that each processor is identical, read: Stirling numbers of the second kind).  This project takes an RL approach, training a controller to output processor assignments for each node based on timing measurements of previous assignments.  This approach is inspired by [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578), which uses an RL-trained controller to output neural network architectures.

## Parallel Computation Graphs
Graphs are input as a list of nodes and an associated list of dependencies.  A node in the computation graph represents some atomic unit of computation with respect to its inputs (e.g. addition, matrix multiply, etc.), while the dependencies specify the inputs to each node.  Three types of graphs are considered in this work: user-defined graphs, random graphs, and parallel permutation graphs.

### User-Defined Graphs
A user can specify an arbitrary directed, acyclic graph in the file `python/main.py`.  An example is given below:
```
input_size = (10, 1)
weights = numpy.eye(10)
node_list = [
    Input(),           #0
    Linear(2*weights), #1
    Linear(3*weights), #2
    Add()              #3
]
                 #1  #2  #3
input_idx_list = [0, 0, [1, 2]]
sess.init_graph(node_list, input_idx_list, input_shape)
```
This code snippet creates a computation graph which takes a column vector of size 10 as input, muliplies it by twice and three times the identity matrix, then returns the sum of resulting column vectors (the output of the final node is always returned as the graph's output).  Note that the dependency list, `input_idx_list` is 1-indexed since we need not account for the input node at index 0.  A generic `Node` class is given in `layers.py` to allow for development of specialized nodes.

### Random Graphs
Random graphs allow for meaningful analysis of the learning method.  To generate a random directed acyclic graph, each node is numbered, and an edge is randomly assigned between two nodes with some probability P, subject to the constraint that the edge points from a lower index node to a higher index node (to ensure the graph remains acyclic).  This can be done purely from the command line by specifying the probability P and the number of nodes:
```
mpiexec -np <num_procs> python3 python/main.py --n_nodes 100 --edge_prob 0.1 <other_args>
```

### Parallel Permutation Graphs
Not all graphs are capable of achieving perfect linear speedup as the number of processors increases.  The parallel permutation graph is a random graph which is capable of perfect speedup with non-trivial learning, allowing for more pointed analysis of the learning method.  This graph type consists of L layers with w nodes per layer, where each node is connected to a node in the subsequent layer via a one-to-one mapping (i.e. a permutation, since w is constant between layers).  This translates to a graph with w distinct "branches", each totally independent from the others, allowing for perfect speedup with up to w processors.  This type of graph can be generated from the command line by specifying L and w:
```
mpiexec -np <num_procs> python3 python/main.py --L 10 --w 16 <other_args>
```
