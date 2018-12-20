import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class Controller(nn.Module):
    def __init__(self, nProc, graph, cl_args):
        super(Controller, self).__init__()
        # Bookkeeping variables
        self.nProc = nProc
        self.output_len = len(graph.node_list) - 1
        self.node_type_dict, self.node_type_list, self.n_node_types = self.parse_nodes(graph.node_list)
        self.input_idx_list = graph.input_idx_list
        self.output_idx_list = graph.output_idx_list
        self.alphabet = np.arange(nProc)
        self.log_probs = []
        self.reward = 0
        self.avg_rew = 0
        self.rew_count = 0
        self.best_reward = -np.inf
        self.best_procs = None
        self.current_procs = None
        self.raw_assignments = None

        # CL Options
        self.lr = cl_args.lr # Float
        self.use_baseline = cl_args.use_baseline # Bool

        # Debugging variables
        self.reward_list = []
        self.loss_list = []

        # Controller architecture variables
        self.hidden_dim = 50
        self.node_embedding_dim = 5 # Encode node type
        self.proc_embedding_dim = 10 # Encode chosen proc (previous output of LSTM)
        self.input_embedding_dim = 10 # Encode inputs to node
        self.output_embedding_dim = 10 # Encode outputs of node
        self.lstm_input_dim = sum([ self.node_embedding_dim,
                                    self.proc_embedding_dim,
                                    self.input_embedding_dim,
                                    self.output_embedding_dim ])
        self.node_embeddings = nn.Embedding(self.n_node_types, self.node_embedding_dim)
        self.proc_embeddings = nn.Embedding(len(self.alphabet) + 1, self.proc_embedding_dim) # +1 for input node
        self.input_embeddings = nn.Sequential(nn.Linear(len(self.node_type_list), 10),
                                              nn.ReLU(), 
                                              nn.Linear(10, 10),
                                              nn.ReLU(),
                                              nn.Linear(10, self.input_embedding_dim))
        self.output_embeddings = nn.Sequential(nn.Linear(len(self.node_type_list), 10),
                                              nn.ReLU(), 
                                              nn.Linear(10, 10),
                                              nn.ReLU(),
                                              nn.Linear(10, self.output_embedding_dim))
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(self.lstm_input_dim, self.hidden_dim)
        self.affine = nn.Linear(self.hidden_dim, len(self.alphabet))
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    # Returns softmax probabilities corresponding to the processor assignment for a single node in the graph
    def forward(self, node_idx, previous_proc):
        # Get embeddings
        node_type = torch.LongTensor([self.node_type_dict[self.node_type_list[node_idx]]])
        node_embeds = self.node_embeddings(node_type).view(1, -1, self.node_embedding_dim) # Flatten tensor to vector

        previous_proc = torch.LongTensor([previous_proc])
        proc_embeds = self.proc_embeddings(previous_proc).view(1, -1, self.proc_embedding_dim) # Flatten tensor to vector

        i = self.multi_hot(self.input_idx_list[node_idx], len(self.node_type_list))
        inputs = torch.from_numpy(i)
        input_embeds = self.input_embeddings(inputs).view(1, -1, self.input_embedding_dim) # Flatten tensor to vector

        o = self.multi_hot(self.output_idx_list[node_idx], len(self.node_type_list))
        outputs = torch.from_numpy(o)
        output_embeds = self.output_embeddings(outputs).view(1, -1, self.output_embedding_dim) # Flatten tensor to vector

        # Run LSTM
        embeds = torch.cat([node_embeds, proc_embeds, input_embeds, output_embeds], dim=2) # Concatenate tensors
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view((1,-1)) # Flatten tensor to vector
        affine_out = self.affine(lstm_out)
        probs = F.softmax(affine_out, dim=1)
        return probs

    # Generates a list specifying the processor assignment for each node in the graph
    def pick_procs(self):
        self.hidden = self.init_hidden()
        self.raw_assignments = []
        out = np.zeros(self.output_len)
        previous_proc = self.nProc # Input node
        for node_idx in range(self.output_len):
            self.raw_assignments.append(previous_proc)
            probs = self(node_idx, previous_proc) # Forward
            m = torch.distributions.Categorical(probs)
            idxVar = m.sample()
            idx = idxVar.data[0]
            proc = self.alphabet[idx]
            self.log_probs.append(m.log_prob(idxVar))
            out[node_idx] = proc
            previous_proc = proc
        self.current_procs = out
        return out

    # Return best processor assignments and associated reward (after training)
    def best(self):
        return self.best_procs, self.best_reward

    # Get external reward signal from graph timing
    def add_reward(self, r):
        self.reward = r
        self.rew_count += 1
        self.avg_rew = self.update_mean(self.avg_rew, r, self.rew_count)
        return self.reward > self.best_reward # If time better than current best, return True so graph can test further

    # If best reward, update best.  Intended for use with a larger number of timing runs for higher accuracy
    def add_potential_best(self, r, proc_list):
        if r > self.best_reward:
            self.best_reward = r
            self.best_procs = proc_list

    # Reinforce algorithm - computes loss from self.rewards, runs backward, steps optimizer
    def reinf(self):
        loss_list = []
        reward = torch.Tensor([self.reward])
        if self.use_baseline:
            reward = reward - self.avg_rew
        for log_prob in self.log_probs:
            loss_list.append(-log_prob * reward)
        self.optimizer.zero_grad()
        loss = torch.cat(loss_list).sum()
        self.loss_list.append(float(loss.data))
        self.reward_list.append(reward)
        loss.backward()
        self.optimizer.step()

    # Initialize LSTM hidden state to all 1's - PyTorch LSTM hidden state is a 2-tuple of Tensors
    def init_hidden(self):
        return (torch.ones(1, 1, self.hidden_dim), torch.ones(1, 1, self.hidden_dim))

    # Converts list of input indices to multi-hot vector
    def multi_hot(self, inputs, length):
        out = np.zeros(length, dtype=np.float32)
        for input_idx in inputs:
            out[input_idx] += 1/len(inputs)
        return out

    # Create dictionary/list of node types for embeddings
    def parse_nodes(self, node_list):
        type_dict = {}
        type_list = []
        n_node_types = 0
        for node in node_list:
            type_list.append(type(node))
            if type(node) not in type_dict:
                type_dict[type(node)] = n_node_types
                n_node_types += 1
        return type_dict, type_list, n_node_types

    # Clear reward and log probs from previous proc list
    def clear_batch(self):
        self.reward = np.inf
        del self.log_probs[:]

    # Incremental update for calculating mean of a sequence
    def update_mean(self, old, new, n):
        return old + (new - old) / n