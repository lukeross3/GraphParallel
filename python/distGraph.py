import numpy as np
from mpi4py import MPI
from layers import Input, Timing, Add

class Graph():
    def __init__(self, node_list, input_idx_list, input_shape, comm, verbose=0):
        # Graph objects
        assert(len(node_list) - 1 == len(input_idx_list))
        self.node_list = node_list
        self.input_shape = input_shape
        self.input_idx_list = input_idx_list
        self.output_idx_list = self.parse_graph(input_idx_list)
        self.proc_list = None
        self.my_nodes = None
        self.verbose = verbose

        # MPI objects
        self.comm = comm
        self.rank = self.comm.rank
        self.size = self.comm.size

    # Get list of outputs for each node from list of inputs for each node
    def parse_graph(self, input_idx_list):
        output_idx_list = [[] for node in self.node_list]
        for i in range(len(input_idx_list)):
            ni = i + 1 # Adjust index to due to input node
            if type(input_idx_list[i]) != list:
                input_idx_list[i] = [input_idx_list[i]]
            if self.node_list[ni] != Input:
                self.node_list[ni].input_dependencies = input_idx_list[i]
            inputs = self.node_list[ni].input_dependencies
            for input_idx in inputs:
                self.node_list[input_idx].output_dependencies.append(ni)
                output_idx_list[input_idx].append(ni)
        return output_idx_list

    # Assign each node to its respective processor
    def proc_init(self, proc_list):
        assert(len(proc_list) == len(self.node_list) - 1)
        self.proc_list = np.insert(proc_list, 0, -1) # Insert -1 for input node
        self.my_nodes = [idx for idx, proc in enumerate(self.proc_list) if proc == self.rank]

        if self.comm.rank == 0 and self.verbose > 2:
            print("Nodes: ", self.node_list)
            print("Inputs: ", self.input_idx_list)
            print("Procs: ", self.proc_list)
        if self.verbose > 2:
            self.comm.Barrier()
        
        if self.comm.rank != self.comm.size-1 and self.verbose > 2:
            print('Rank: ' + str(self.rank))
            print(self.my_nodes)
            for i, node in enumerate(self.node_list):
                print("\t", i, node.input_dependencies, node.output_dependencies)
        if self.verbose > 2:
            self.comm.Barrier()

    # Run the computation graph on input x
    def forward(self, x):
        for node_idx in self.my_nodes:
            # If no one needs output, skip!
            output_dependencies = self.node_list[node_idx].output_dependencies
            if len(output_dependencies) == 0:
                continue

            # Get inputs
            input_dependencies = self.node_list[node_idx].input_dependencies
            reqs = []
            inputs = [np.empty(self.input_shape) for i in range(len(input_dependencies))]
            for i in range(len(input_dependencies)):
                input_node = input_dependencies[i]
                if input_node in self.my_nodes: # Input from same processor, no need for MPI
                    inputs[i] = self.node_list[input_node].output
                elif input_node == 0: # Input layer
                    inputs[i] = x
                else: # Post a receive for data from other proc (non-blocking)
                    reqs.append(self.comm.Irecv(inputs[i], source=self.proc_list[input_node]))
                    if self.verbose > 2:
                        print("Post Recv: ", input_node, "-->", node_idx)
            MPI.Request.Waitall(reqs) # Wait until all inputs available

            # Run forward computation
            output = self.node_list[node_idx].forward(inputs)

            # Send output to next nodes
            for next_node in output_dependencies:
                next_node_n_dependencies = len(self.node_list[next_node].output_dependencies)
                if next_node not in self.my_nodes and next_node_n_dependencies > 0: # Only send if to a different node
                    self.comm.Isend(output, dest=self.proc_list[next_node])
                    if self.verbose > 2:
                        print("Post Send: ", node_idx, "-->", next_node)

        # Return last value computed if stored on this processor, else None
        if len(self.node_list) - 1 in self.my_nodes:
            return self.node_list[-1].output
        return None

    # Compute the runtime of the computation graph averaged over 'nIter' runs
    def time(self, nIter):
        t = 0
        x = np.ones(self.input_shape)
        for i in range(nIter):
            self.comm.Barrier()
            t0 = MPI.Wtime()
            y = self.forward(x)
            self.comm.Barrier()
            t1 = MPI.Wtime()
            t += (t1-t0)
            self.reset_nodes()
        return t/nIter

    # Reset each node in the graph (deletes any stored data in nodes)
    def reset_nodes(self):
        for i in range(len(self.node_list)):
            self.node_list[i].reset()

# Note: numpy random seed must be set to be the same for all processors so they generate the same
#       random graph.  Handled in main.py
class RandomGraph(Graph):
    def __init__(self, n_nodes, edge_prob, comm, input_shape=(300,1), t=0.0001, verbose=0):
        self.edge_prob = edge_prob
        node_list = [Input()] + [Timing(t=t, input_shape=input_shape) for i in range(n_nodes-1)]
        input_idx_list = [[] for i in range(n_nodes-1)]
        for input_idx in range(n_nodes-1):
            for output_idx in range(input_idx, n_nodes-1):
                edge_exists = np.random.choice([True, False], p=[edge_prob, 1-edge_prob])
                if edge_exists:
                    input_idx_list[output_idx].append(input_idx)
        super().__init__(node_list, input_idx_list, input_shape, comm, verbose=verbose)

# Graph with L layers of width w which can achieve perfect scaling nontrivially
class ParGraph(Graph):
    def __init__(self, L, w, comm, input_shape=(300,1), t=0.0001, verbose=0, permute=True):
        self.L = L
        self.w = w
        node_list = [Input()] + [Timing(t=t, input_shape=input_shape) for i in range(L * w)] + [Add()]
        input_idx_list = [[] for i in range(L * w + 1)]

        # First layer all from input node
        for node_idx in range(1, w+1):
            input_idx_list[node_idx].append(0)

        # Each subsequent layer gets its input from 1 random node from previous layer
        for layer in range(L-1):
            perm = np.random.permutation(w)
            if not permute:
                perm = np.arange(w)
            for current_w in range(w):
                input_node_idx = layer*w + current_w + 1
                output_node_idx = (layer+1)*w + perm[current_w] + 1
                input_idx_list[output_node_idx].append(input_node_idx)

        # Final node adds everything from output layer
        input_idx_list[-1] = list(range(len(node_list)-w-1, len(node_list)-1))
        super().__init__(node_list, input_idx_list, input_shape, comm, verbose=verbose)