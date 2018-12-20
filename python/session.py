import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from layers import Input, Timing, Linear, Add
from distGraph import Graph, RandomGraph, ParGraph
from controller import Controller
from utils import window_avg, fit

class Session():
    def __init__(self, comm, cl_args):
        self.comm = comm
        self.cl_args = cl_args
        self.is_graph_random = False
        self.is_graph_par = False
        self.graph = None

    def init_random_graph(self, input_shape=(3000,1), t=0.0001):
        self.is_graph_random = True
        self.graph = RandomGraph(self.cl_args.n_nodes, self.cl_args.edge_prob, self.comm, input_shape=input_shape, t=t, verbose=self.cl_args.verbose)

    def init_par_graph(self, input_shape=(3000,1), permute=True, t=0.0001):
        self.is_graph_par = True
        self.graph = ParGraph(self.cl_args.L, self.cl_args.w, self.comm, input_shape=input_shape, t=t, verbose=self.cl_args.verbose, permute=permute)

    def init_graph(self, node_list, input_idx_list, input_shape):
        self.graph = Graph(node_list, input_idx_list, input_shape, self.comm, verbose=self.cl_args.verbose)

    # Test 'n_iter' random processor assignments, return average time
    def random_proc_baseline(self, n_iter):
        t = 0
        for i in range(n_iter):
            proc_list = np.random.randint(self.comm.size, size=len(self.graph.node_list)-1).astype(float)
            self.graph.proc_init(proc_list)
            t += self.graph.time(self.cl_args.timing_runs)
        return t / n_iter

    # Runs the learning procedure
    # Prereqs: must have initialized session graph
    def learn_assignments(self):
        # Initialize the controller on process 0
        if self.comm.rank == 0:
            controller = Controller(self.comm.size, self.graph, self.cl_args)
            times = []

        # Train Controller
        for i in range(self.cl_args.n_iter):
            # Initialize Processor Assignments
            if self.comm.rank == 0:
                proc_list = controller.pick_procs()
            else:
                proc_list = np.empty(len(self.graph.node_list)-1)
            self.comm.Bcast(proc_list, root=0)
            self.graph.proc_init(proc_list)

            # Perform controller update
            t = self.graph.time(self.cl_args.timing_runs)
            r = -t # Reward signal is -time taken
            potential_best = None
            if self.comm.rank == 0:
                if self.cl_args.verbose > 1:
                    print(str(i) + '/' + str(self.cl_args.n_iter) + ':')
                    print(t)
                    print(proc_list)
                times.append(t)
                potential_best = controller.add_reward(r)
                controller.reinf()
                controller.clear_batch()

            # Check for potential best assignment
            potential_best = self.comm.bcast(potential_best, root=0)
            if potential_best:
                t_acc = self.graph.time(10 * self.cl_args.timing_runs) # 10x more timing runs for accuracy
                r_acc = -t_acc # Reward signal is -time taken
                if self.comm.rank == 0:
                    controller.add_potential_best(r_acc, proc_list)

        # Get best performing assignments after done training
        if self.comm.rank == 0:
            best_proc_list, best_reward = controller.best()
        else:
            best_proc_list = np.empty(len(self.graph.node_list)-1)
        self.comm.Bcast(best_proc_list, root=0)
        self.graph.proc_init(best_proc_list)
        b_t = self.random_proc_baseline(100 * self.cl_args.timing_runs)
        if self.comm.rank == 0:
            if self.cl_args.verbose > 1:
                print()
            if self.cl_args.verbose > 0:
                print("Number of Processors:       " + str(self.comm.size))
                print("Best Processor Assignment:  " + str(best_proc_list))
                print("Best Assignment Runtime:    " + str(-best_reward))
                print("Random Assignment Baseline: " + str(b_t))
            else:
                print(self.comm.size)
                print(-best_reward)
                print(b_t)

        # Generate learning plot
        if self.comm.rank == 0:
            self.save_plot(times, b_t=b_t)

    # Generate and save learning plot as 'fig.png'
    def save_plot(self, times, b_t=None):
        plt.plot(times, label='Times')
        x, y = window_avg(times, 50) # Windowed average for smoothing
        plt.plot(x, y, label='Windowed Avg')
        x_r, y_r = fit(times) # Linear best fit line
        plt.plot(x_r, y_r, label='Linear Fit')
        if b_t is not None:
            plt.plot([0,len(times)], [b_t, b_t], label='Random Proc Baseline')
        plt.xlabel('Iteration')
        plt.ylabel('Time (s)')
        title = ''
        if self.is_graph_random:
            title = 'Random Graph: N=' + str(len(self.graph.node_list)) + \
            ', P=' + str(self.graph.edge_prob) + \
            ', p=' + str(self.comm.size) + \
            ', lr=' + str(self.cl_args.lr) + \
            ', seed=' + str(self.cl_args.seed)
        elif self.is_graph_par:
            title = 'Par Graph: L=' + str(self.graph.L) + \
            ', w=' + str(self.graph.w) + \
            ', p=' + str(self.comm.size) + \
            ', lr=' + str(self.cl_args.lr) + \
            ', seed=' + str(self.cl_args.seed)
        else:
            title = 'User Graph: N=' + str(len(self.graph.node_list)) + \
            ', p=' + str(self.comm.size) + \
            ', lr=' + str(self.cl_args.lr) + \
            ', seed=' + str(self.cl_args.seed)
        plt.title(title)
        plt.legend(loc='best')
        plt.savefig('fig.png')

    def debug_random_graph(self, n_nodes, edge_prob, proc_list, n_iter, input_shape=(300,1), t=0.0001):
        # Example to debug random graphs
        node_list = [Input()] + [Timing(input_shape=input_shape, t=t) for i in range(n_nodes)]
        x = np.ones(input_shape)
        for i in range(n_iter):
            graph = RandomGraph(n_nodes, edge_prob, self.comm, verbose=True)
            graph.proc_init(proc_list)
            y = graph.forward(x)
            if comm.rank == 0:
                print()
            time.sleep(0.05)

    def parallel_test(self, permute_edges=True, assignment_eff='max', input_shape=(3000,1), t=0.0001):
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
        # Graph with 4 parallel lines
        #                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13
        input_idx_list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, [9,10,11,12]]
        if permute_edges:
            # Graph with 4 parallel lines, but nontrivial ordering
            #                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13
            input_idx_list = [0, 0, 0, 0, 2, 3, 1, 4, 8, 7, 5, 6, [9,10,11,12]]

        # Build graph
        graph = Graph(node_list, input_idx_list, input_shape, self.comm)

        # Choose and initialize processor assignment
        proc_list = np.array([0,1,2,3,1,2,0,3,3,0,2,1,0]) # Maximally efficient
        if assignment_eff == 'med':
            proc_list = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0]) # Parallel, but wasteful sends
        elif assignment_eff == 'min':
            proc_list = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0]) # Maximally inefficient
        elif assignment_eff != 'max':
            print("Error: unrecognized assignment efficiency: " + str(assignment_eff))
        graph.proc_init(proc_list)

        # Timing
        t = graph.time(self.cl_args.timing_runs)
        if self.comm.rank == 0:
            print(t)