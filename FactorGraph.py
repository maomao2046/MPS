import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy as cp


class Graph:

    def __init__(self, number_of_nodes=None):
        self.node_count = 0
        self.factors = {}
        self.nodes = {}
        self.factors_count = 0
        self.messeges_n2f = None
        self.messeges_f2n = None

    def add_node(self, alphabet_size, name):
        self.nodes[name] = [alphabet_size, set()]
        self.node_count += 1

    def add_factor(self, node_neighbors, tensor):
        factor_name = 'f' + str(self.factors_count)
        for i in range(len(node_neighbors)):
            if node_neighbors[i] not in self.nodes.keys():
                raise IndexError('Tried to factor non exciting node')
            if tensor.shape[i] != self.nodes[node_neighbors[i]][0]:
                raise IndexError('There is a mismatch between node alphabet and tensor size')
            self.nodes[node_neighbors[i]][1].add(factor_name)
        self.factors[factor_name] = [node_neighbors, tensor]
        self.factors_count += 1

    def sum_product(self, t_max, epsilon):
        factors = self.factors
        nodes = self.nodes
        node_count = self.node_count
        factor_count = self.factors_count

        node2factor = {}
        factor2node = {}
        for n in nodes.keys():
            node2factor[n] = {}
            alphabet = nodes[n][0]
            for f in nodes[n][1]:
                node2factor[n][f] = np.ones(alphabet, dtype=complex) / alphabet

        for f in factors.keys():
            factor2node[f] = {}
            for n in factors[f][0]:
                alphabet = nodes[n][0]
                factor2node[f][n] = np.ones(alphabet, dtype=complex) / alphabet

        for t in range(t_max):



        self.messeges_n2f = node2factor
        self.messeges_f2n = factor2node
