import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy as cp

# Non complex Loopy Belief Propagation


class Graph:

    def __init__(self, number_of_nodes=None):
        self.node_count = 0
        self.factors = {}
        self.nodes = {}
        self.factors_count = 0
        self.messages_n2f = None
        self.messages_f2n = None
        self.node_belief = None
        self.factor_belief = None
        self.DENFGbeliefs = None

    def add_node(self, alphabet_size, name):
        self.nodes[name] = [alphabet_size, set()]
        self.node_count += 1

    def add_factor(self, node_neighbors, tensor):
        factor_name = 'f' + str(self.factors_count)
        for n in node_neighbors.keys():
            if n not in self.nodes.keys():
                raise IndexError('Tried to factor non exciting node')
            if tensor.shape[node_neighbors[n]] != self.nodes[n][0]:
                raise IndexError('There is a mismatch between node alphabet and tensor size')
            self.nodes[n][1].add(factor_name)
        self.factors[factor_name] = [node_neighbors, tensor]
        self.factors_count += 1

    def broadcasting(self, message, idx, tensor):
        new_shape = np.ones(len(tensor.shape), dtype=np.int)
        new_shape[idx] = message.shape[0]
        return np.reshape(message, new_shape)

    def sum_product(self, t_max, epsilon):
        factors = self.factors
        nodes = self.nodes
        node_count = self.node_count
        factor_count = self.factors_count
        counter = 0

        node2factor = {}
        factor2node = {}
        for n in nodes.keys():
            node2factor[n] = {}
            alphabet = nodes[n][0]
            for f in nodes[n][1]:
                node2factor[n][f] = np.ones(alphabet, dtype=float) / alphabet
                #if n == 'n0':
                 #   node2factor[n][f] = np.array([0.4, 0.6])

        for f in factors.keys():
            factor2node[f] = {}
            for n in factors[f][0]:
                alphabet = nodes[n][0]
                factor2node[f][n] = np.ones(alphabet, dtype=float) / alphabet


        for t in range(t_max):
            for n in nodes.keys():
                for f in nodes[n][1]:
                    neighbor_factors = cp.deepcopy(nodes[n][1])
                    neighbor_factors.remove(f)
                    temp_message = np.ones(node2factor[n][f].shape, dtype=float)
                    for item in neighbor_factors:
                        temp_message *= factor2node[item][n]
                    node2factor[n][f] = cp.copy(temp_message)
                    node2factor[n][f] /= np.sum(node2factor[n][f])

            for f in factors.keys():
                for n in factors[f][0].keys():
                    tensor = cp.deepcopy(factors[f][1])
                    tensor_idx = range(len(tensor.shape))
                    neighbor_nodes = cp.deepcopy(factors[f][0].keys())
                    message_idx = [factors[f][0][n]]
                    neighbor_nodes.remove(n)
                    for item in neighbor_nodes:
                        tensor *= self.broadcasting(node2factor[item][f], factors[f][0][item], tensor)
                    factor2node[f][n] = np.einsum(tensor, tensor_idx, message_idx)
                    factor2node[f][n] /= np.sum(factor2node[f][n])

        self.messages_n2f = node2factor
        self.messages_f2n = factor2node

    def beliefs(self):
        self.node_belief = {}
        for n in self.nodes:
            self.node_belief[n] = np.ones(self.nodes[n][0], dtype=float)
            for f in self.nodes[n][1]:
                self.node_belief[n] *= self.messages_f2n[f][n]
            self.node_belief[n] /= np.sum(self.node_belief[n])




