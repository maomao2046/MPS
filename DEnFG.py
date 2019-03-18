import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy as cp
from scipy.stats import unitary_group


class Graph:

    def __init__(self, number_of_nodes=None):
        self.node_count = 0
        self.factors = {}
        self.nodes_order = []
        self.node_indices = {}
        self.nodes = {}
        self.factors_count = 0
        self.messages_n2f = None
        self.messages_f2n = None
        self.node_partition = None
        self.factor_partition = None

    def add_node(self, alphabet_size, name):
        self.nodes[name] = [alphabet_size, set()]
        self.nodes_order.append(name)
        self.node_indices[name] = self.node_count
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
        idx = [2 * idx, 2 * idx + 1]
        new_shape = np.ones(len(tensor.shape), dtype=np.int)
        new_shape[idx] = message.shape
        return np.reshape(message, new_shape)

    def tensor_broadcasting(self, tensor, idx, master_tensor):
        new_shape = np.ones(len(master_tensor.shape), dtype=np.int)
        new_shape[idx] = tensor.shape
        return np.reshape(tensor, new_shape)

    def pd_mat_init(self, alphabet):
        eigenval = np.eye(alphabet)
        for i in range(alphabet):
            eigenval[i, i] = np.random.rand()
        eigenval /= np.trace(eigenval)
        unitary = unitary_group.rvs(alphabet)
        pd = np.matmul(np.transpose(np.conj(unitary)), np.matmul(eigenval, unitary))
        return pd

    def make_super_tensor(self, tensor):
        tensor_idx1 = np.array(range(len(tensor.shape)))
        tensor_idx2 = cp.copy(tensor_idx1) + len(tensor_idx1)
        super_tensor_idx_shape = []
        for i in range(len(tensor_idx1)):
            super_tensor_idx_shape.append(tensor_idx1[i])
            super_tensor_idx_shape.append(tensor_idx2[i])
        super_tensor = np.einsum(tensor, tensor_idx1, np.conj(tensor), tensor_idx2, super_tensor_idx_shape)
        return super_tensor

    def sum_product(self, t_max, epsilon=None):
        factors = self.factors
        nodes = self.nodes
        node2factor = {}
        factor2node = {}
        for n in nodes.keys():
            node2factor[n] = {}
            alphabet = nodes[n][0]
            for f in nodes[n][1]:
                node2factor[n][f] = self.pd_mat_init(alphabet)

        for f in factors.keys():
            factor2node[f] = {}
            for n in factors[f][0]:
                alphabet = nodes[n][0]
                factor2node[f][n] = self.pd_mat_init(alphabet)

        for t in range(t_max):
            for n in nodes.keys():
                alphabet = nodes[n][0]
                for f in nodes[n][1]:
                    neighbor_factors = cp.deepcopy(nodes[n][1])
                    neighbor_factors.remove(f)
                    temp_message = np.ones((alphabet, alphabet), dtype=complex)
                    for item in neighbor_factors:
                        temp_message *= factor2node[item][n]
                    if not neighbor_factors:
                        continue
                    else:
                        node2factor[n][f] = cp.copy(temp_message)
                        node2factor[n][f] /= np.trace(node2factor[n][f])

            for f in factors.keys():
                for n in factors[f][0].keys():
                    tensor = cp.deepcopy(factors[f][1])
                    super_tensor = self.make_super_tensor(tensor)
                    neighbor_nodes = cp.deepcopy(factors[f][0].keys())
                    message_idx = [2 * factors[f][0][n], 2 * factors[f][0][n] + 1]
                    neighbor_nodes.remove(n)
                    for item in neighbor_nodes:
                        super_tensor *= self.broadcasting(node2factor[item][f], factors[f][0][item], super_tensor)
                    factor2node[f][n] = np.einsum(super_tensor, range(len(super_tensor.shape)), message_idx)
                    factor2node[f][n] /= np.trace(factor2node[f][n])
        self.messages_n2f = node2factor
        self.messages_f2n = factor2node

    def calc_node_partition(self):
        nodes = self.nodes
        messages = self.messages_f2n
        self.node_partition = {}
        keys = nodes.keys()
        for n in keys:
            alphabet = nodes[n][0]
            temp = np.ones((alphabet, alphabet), dtype=complex)
            for f in nodes[n][1]:
                temp *= messages[f][n]
            self.node_partition[n] = np.sum(temp)

    def calc_factor_partition(self):
        factors = self.factors
        messages = self.messages_n2f
        self.factor_partition = {}
        keys = factors.keys()
        for f in keys:
            super_tensor = self.make_super_tensor(cp.deepcopy(factors[f][1]))
            neighbors = factors[f][0]
            for n in neighbors.keys():
                super_tensor *= self.broadcasting(messages[n][f], neighbors[n], super_tensor)
            self.factor_partition[f] = np.sum(super_tensor)

    def bethe_partition(self):
        bethe = 1
        for f in self.factor_partition:
            bethe *= self.factor_partition[f]
        for n in self.node_partition:
            if np.abs(self.node_partition[n]) < 1e-10:
                raise IndexError('The partition of node' + str(n) + 'is 0')
            bethe *= self.node_partition[n]
        return bethe

    def calc_partition(self):
        node_count = self.node_count
        shape = []
        for n in self.nodes_order:
            shape.append(self.nodes[n][0])
        master_tensor = np.ones(shape, dtype=np.complex128)
        super_master_tensor = self.make_super_tensor(master_tensor)
        for f in self.factors:
            neighbors = self.factors[f][0].keys()
            idx = []
            for n in self.nodes_order:
                if n in neighbors:
                    idx.append(self.node_indices[n])
            super_master_tensor *= self.make_super_tensor(self.tensor_broadcasting(self.factors[f][1], idx, master_tensor))
        return np.sum(super_master_tensor)

