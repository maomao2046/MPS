import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy as cp


class Graph:

    def __init__(self, number_of_nodes=None):
        self.node_count = 0
        self.factors = {}
        self.nodes = []
        self.factors_count = 0
        self.number_of_nodes = number_of_nodes

    def add_node(self, alphabet_size, node_name=None):
        node_name = 'n' + str(self.node_count)
        self.nodes.append([node_name, alphabet_size, set()])
        self.node_count += 1
        return self.node_count

    def broadcasting(self, factor, nodes):
        new_shape = np.ones(self.number_of_nodes, dtype=np.int)
        new_shape[nodes] = np.shape(factor)
        return np.reshape(factor, new_shape)

    def add_factor(self, factor_nodes, boltzmann_factor, factor_name=None):
        factor_name = 'F'
        for i in range(len(factor_nodes)):
            factor_name += ',' + str(factor_nodes[i])
        for i in range(len(factor_nodes)):
            if factor_nodes[i] > self.node_count:
                raise IndexError('Tried to factor non exciting node')
            self.nodes[factor_nodes[i]][2].add(factor_name)
        self.factors_count += 1
        #self.factors[factor_name] = [factor_nodes, self.broadcasting(np.exp( boltzmann_factor), factor_nodes)]
        self.factors[factor_name] = [factor_nodes, self.broadcasting(boltzmann_factor, factor_nodes)]

    def vis_graph(self, flag):
        L = np.int(np.sqrt(self.node_count))
        node_keys = []
        for i in range(self.node_count):
            node_keys.append(self.nodes[i][0])
        factor_keys = self.factors.keys()
        G = nx.Graph()
        G.add_nodes_from(node_keys)
        G.add_nodes_from(factor_keys)
        node_pos = {}
        factor_pos = {}
        field_pos = {}
        pos = {}
        i = 0
        j = 0
        if flag == 'grid':
            for n in range(L):
                for m in range(L):
                    node_name = self.nodes[i][0]
                    node_name_right = self.nodes[n * L + np.mod(i + 1, L)][0]
                    node_name_down = self.nodes[np.mod(i + L, L * L)][0]
                    node_pos[node_name] = [n, m]
                    pos[node_name] = [n, m]
                    factor_name_right = 'F,' + str(n * L + m) + ',' + str(n * L + np.mod(m + 1, L))
                    factor_name_down = 'F,' + str(n * L + m) + ',' + str(np.mod(n + 1, L) * L + m)
                    factor_name_field = 'F,' + str(i)
                    factor_pos[factor_name_right] = [n, m + 0.5]
                    factor_pos[factor_name_down] = [n + 0.5, m]
                    factor_pos[factor_name_field] = [n + 0.25, m + 0.25]
                    pos[factor_name_right] = [n, m + 0.5]
                    pos[factor_name_down] = [n + 0.5, m]
                    pos[factor_name_field] = [n + 0.25, m + 0.25]
                    G.add_edge(node_name, factor_name_right)
                    G.add_edge(node_name_right, factor_name_right)
                    G.add_edge(node_name, factor_name_down)
                    G.add_edge(node_name_down, factor_name_down)
                    G.add_edge(node_name, factor_name_field)
                    i += 1

            node_sub = G.subgraph(node_keys)
            factor_sub = G.subgraph(factor_keys)
            plt.figure()
            nx.draw_networkx(node_sub, pos=node_pos, node_color='b', node_shape='o', node_size=200)
            nx.draw_networkx(factor_sub, pos=factor_pos, node_color='r', node_shape='s', node_size=300)
            nx.draw_networkx_edges(G, pos=pos)
            plt.show()

        if flag == 'no grid':
            for item in range(self.node_count):
                temp = cp.copy(self.nodes[item][0])
                node_pos[temp] = [i, j]
                pos[temp] = [i, j]
                i += 1
                for key in self.nodes[item][2]:
                    G.add_edge(temp, key)
            i = 0
            j += 1
            for item in self.factors:
                factor_pos[item] = [i, j]
                pos[item] = [i, j]
                i += 1
            node_sub = G.subgraph(node_keys)
            factor_sub = G.subgraph(factor_keys)
            plt.figure()
            nx.draw_networkx(node_sub, pos=node_pos, node_color='b', node_shape='o', node_size=200)
            nx.draw_networkx(factor_sub, pos=factor_pos, node_color='r', node_shape='s', node_size=500)
            nx.draw_networkx_edges(G, pos=pos)
            plt.show()

        if flag == 'no vis':
            return

    def exact_partition(self):
        alphabet = np.zeros(self.node_count, dtype=np.int)
        for i in range(self.node_count):
            alphabet[i] = self.nodes[i][1]
        z = np.ones(np.array(alphabet), dtype=complex)
        for item in self.factors:
            z *= self.factors[item][1]
        z = np.sum(z)
        return z

    def sum_product(self, t_max, epsilon):

        factors = self.factors
        nodes = self.nodes
        node2factor = []
        factor2node = {}
        node_belief = []
        factor_beliefs = {}
        counter = 0


        '''
            Initialization of messages and beliefs
        '''
        for item in factors:
            factor_beliefs[item] = [cp.deepcopy(factors[item][1]) / np.sum(cp.deepcopy(factors[item][1]))]

        for i in range(self.node_count):
            alphabet = nodes[i][1]
            node2factor.append({})
            node_belief.append([])
            node_belief[i].append(np.ones(alphabet, dtype=complex) / alphabet)

            for item in nodes[i][2]:
                node2factor[i][item] = np.ones(alphabet, dtype=complex) / alphabet
                node2factor[i][item] = self.broadcasting(node2factor[i][item], np.array([i]))

        '''
            Preforming sum product iterations
        '''
        for t in range(t_max):

            for item in factors:
                neighbors_nodes = cp.deepcopy(factors[item][0])
                factor2node[item] = {}
                temp_factor = cp.deepcopy(factors[item][1])
                for i in range(len(neighbors_nodes)):
                    temp = cp.deepcopy(factors[item][1])  # temp is holding the message until it is ready
                    for j in range(len(neighbors_nodes)):
                        if neighbors_nodes[j] == neighbors_nodes[i]:
                            continue
                        else:
                            temp *= node2factor[neighbors_nodes[j]][item]
                    temp_factor *= node2factor[neighbors_nodes[i]][item]  # temp_factor is holding the factor belief until it is ready
                    temp = np.einsum(temp, range(self.node_count), [neighbors_nodes[i]])
                    factor2node[item][neighbors_nodes[i]] = np.reshape(temp / np.sum(temp), nodes[neighbors_nodes[i]][1])
                factor_beliefs[item].append(np.array(temp_factor / np.sum(temp_factor)))
                if t > 6 and np.abs(np.sum(np.abs(factor_beliefs[item][t] - factor_beliefs[item][t - 1]))) < epsilon:
                    counter += 1

            for i in range(self.node_count):
                alphabet = nodes[i][1]
                temp = self.broadcasting(np.ones(alphabet, dtype=complex) / alphabet, np.array([i]))
                for item in nodes[i][2]:
                    node2factor[i][item] = np.ones(alphabet, dtype=complex) / alphabet
                    node2factor[i][item] = self.broadcasting(node2factor[i][item], np.array([i]))
                    for object in nodes[i][2]:
                        if object == item:
                            continue
                        else:
                            node2factor[i][item] *= self.broadcasting(np.array(factor2node[object][i]), np.array([i]))
                    node2factor[i][item] /= np.sum(node2factor[i][item], axis=i)
                    temp *= self.broadcasting(np.array(factor2node[item][i]), np.array([i]))
                node_belief[i].append(np.reshape(temp, [alphabet]) / np.sum(temp))
                if t > 6 and np.abs(np.sum(np.abs(node_belief[i][t] - node_belief[i][t - 1]))) < epsilon:
                    counter += 1
            if counter == self.factors_count + self.node_count:
                return np.array(node_belief), factor_beliefs, t + 1
            else:
                counter = 0
        return np.array(node_belief), factor_beliefs, t_max

    def mean_field_approx_to_F(self, node_beliefs):
        energy = 0
        entropy = 0
        for item in self.factors:
            temp_energy = -np.log(cp.deepcopy(self.factors[item][1]))
            neighbors = cp.deepcopy(self.factors[item][0])
            for i in range(len(neighbors)):
                temp_energy *= self.broadcasting(node_beliefs[neighbors[i]], np.array([neighbors[i]]))
            energy += np.abs(np.sum(temp_energy))
        for i in range(self.node_count):
            entropy -= np.abs(np.dot(node_beliefs[i], np.log(node_beliefs[i])))  # think about the absolute value here
        F_approx = energy - entropy
        return F_approx

    def bethe_approx_to_F(self, node_beliefs, factor_beliefs):
        energy = 0
        entropy = 0
        for item in self.factors:
            energy += np.absolute(np.sum(- factor_beliefs[item] * np.log(self.factors[item][1])))
            entropy += np.abs(np.sum(- factor_beliefs[item] * np.log(factor_beliefs[item])))  # and here
        for i in range(self.node_count):
            d = len(self.nodes[i][2])
            entropy -= np.abs((1 - d) * np.dot(node_beliefs[i], np.log(node_beliefs[i])))  # and here
        F_bethe_approx = energy - entropy
        return F_bethe_approx

    def SumProduct(self):
        factors = self.factors
        nodes = self.nodes
        node2factor = {}
        factor2node = {}
        counter = 0
        for item in range(self.node_count):
            node2factor[item] = {}
