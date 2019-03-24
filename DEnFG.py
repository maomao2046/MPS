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
        self.node_belief = None
        self.factor_belief = None
        self.factors_count = 0
        self.messages_n2f = None
        self.messages_f2n = None
        self.node_partition = None
        self.factor_partition = None
        self.all_messages = None

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
        eigenval = np.eye(alphabet) / alphabet
        #for i in range(alphabet):
        #    eigenval[i, i] = np.random.rand()
        #eigenval /= np.trace(eigenval)
        #unitary = unitary_group.rvs(alphabet)
        #pd = np.matmul(np.transpose(np.conj(unitary)), np.matmul(eigenval, unitary))
        return eigenval

    def make_super_tensor(self, tensor):
        tensor_idx1 = np.array(range(len(tensor.shape)))
        tensor_idx2 = cp.copy(tensor_idx1) + len(tensor_idx1)
        super_tensor_idx_shape = []
        for i in range(len(tensor_idx1)):
            super_tensor_idx_shape.append(tensor_idx1[i])
            super_tensor_idx_shape.append(tensor_idx2[i])
        super_tensor = np.einsum(tensor, tensor_idx1, np.conj(tensor), tensor_idx2, super_tensor_idx_shape)
        return super_tensor

    def sum_product(self, t_max, epsilon):
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
        self.init_save_messages()

        for t in range(t_max):
            old_messages_f2n = factor2node
            old_messages_n2f = node2factor
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
            self.save_messages(node2factor, factor2node)
        self.messages_n2f = node2factor
        self.messages_f2n = factor2node

    def check_converge(self, n2f_old, f2n_old, n2f_new, f2n_new, epsilon):
        diff = 0
        for n in n2f_old:
            for f in n2f_old[n]:
                diff += np.sum(np.abs(n2f_old[n][f] - n2f_new[n][f]))
        for f in f2n_old:
            for n in f2n_old[f]:
                diff += np.sum(np.abs(f2n_old[f][n] - f2n_new[f][n]))
        if diff < epsilon:
            return 1
        else:
            return 0

    def calc_node_partition(self):
        nodes = self.nodes
        messages = self.messages_f2n
        #messages = self.messages_n2f
        self.node_partition = {}
        keys = nodes.keys()
        for n in keys:
            d = len(nodes[n][1])
            alphabet = nodes[n][0]
            temp = np.ones((alphabet, alphabet), dtype=complex)
            for f in nodes[n][1]:
                temp *= messages[f][n]
                #temp *= messages[n][f]
            temp = temp ** (d - 1)
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
                raise IndexError('The partition of node' + n + 'is 0')
            bethe *= self.node_partition[n]
        return bethe

    def calc_partition(self):
        shape = []
        for n in self.nodes_order:
            shape.append(self.nodes[n][0])
        master_tensor = np.ones(shape, dtype=np.complex128)
        super_master_tensor = self.make_super_tensor(master_tensor)
        for f in self.factors:
            neighbors = self.factors[f][0]
            idx = np.zeros(len(self.factors[f][1].shape), dtype=int)
            for n in self.nodes_order:
                if n in neighbors:
                    idx[neighbors[n]] = self.node_indices[n]
            idx = np.ndarray.tolist(idx)
            super_master_tensor *= self.make_super_tensor(self.tensor_broadcasting(self.factors[f][1], idx, master_tensor))
        return np.sum(super_master_tensor)

    def save_messages(self, n2f, f2n):
            for n in self.nodes:
                for f in self.nodes[n][1]:
                    self.all_messages[n][f].append(n2f[n][f])
            for f in self.factors:
                for n in self.factors[f][0]:
                    self.all_messages[f][n].append(f2n[f][n])

    def init_save_messages(self):
        self.all_messages = {}
        for n in self.nodes:
            self.all_messages[n] = {}
            for f in self.nodes[n][1]:
                self.all_messages[n][f] = []
        for f in self.factors:
            self.all_messages[f] = {}
            for n in self.factors[f][0]:
                self.all_messages[f][n] = []

    def calc_node_belief(self):
        self.node_belief = {}
        nodes = self.nodes
        messages = self.messages_f2n
        keys = nodes.keys()
        for n in keys:
            alphabet = nodes[n][0]
            temp = np.ones((alphabet, alphabet), dtype=complex)
            for f in nodes[n][1]:
                temp *= messages[f][n]
            self.node_belief[n] = temp

    def calc_factor_belief(self):
        self.factor_belief = {}
        factors = self.factors
        messages = self.messages_n2f
        keys = factors.keys()
        for f in keys:
            super_tensor = self.make_super_tensor(cp.deepcopy(factors[f][1]))
            neighbors = factors[f][0]
            for n in neighbors.keys():
                super_tensor *= self.broadcasting(messages[n][f], neighbors[n], super_tensor)
            self.factor_belief[f] = super_tensor

    def bethe_partition2(self):
        shape = []
        for n in self.nodes_order:
            shape.append(self.nodes[n][0])
        master_tensor = np.ones(shape, dtype=np.complex128)
        # not true
        '''
        for f in self.factor_belief:
            print(np.sum(self.factor_belief[f]))
            z_bethe *= np.sum(self.factor_belief[f])
        for n in self.node_belief:
            print(np.sum(self.node_belief[n] ** (len(self.nodes[n][1]) - 1)))
            z_bethe /= np.sum(self.node_belief[n] ** (len(self.nodes[n][1]) - 1))
            '''
        E_b = self.calc_bethe_energy()
        H_b = self.calc_bethe_entropy()
        F_b = E_b - H_b
        z_bethe = -np.log(F_b)
        return z_bethe

    def calc_bethe_energy(self):
        E_b = 0
        for f in self.factor_belief:
            E_b += np.sum(self.factor_belief[f] * np.log(self.make_super_tensor(self.factors[f][1])))
        return -E_b
    def calc_bethe_entropy(self):
        H_b = 0
        for f in self.factor_belief:
            H_b -= np.sum(self.factor_belief[f] * np.log(self.factor_belief[f]))
        for n in self.node_belief:
            H_b += (len(self.nodes[n][1]) - 1) * np.sum(self.node_belief[n] * np.log(self.node_belief[n]))
        return  H_b








