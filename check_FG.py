import numpy as np
import matplotlib.pyplot as plt
import matrix_product_state as mps
import copy as cp
import wavefunction as wf
import FactorGraph as fg

def ising_factor(k):
    return np.exp(-np.array([[k , -k], [-k, k]]))

k = 0.3
graph = fg.Graph()
graph.add_node(2, 'n0')
graph.add_node(2, 'n1')
graph.add_node(2, 'n2')

graph.add_factor({'n0': 0, 'n1': 1}, np.exp(-np.array([[1e6, 1e6 + 0.j], [0, 1e6]])))
graph.add_factor({'n1': 0, 'n2': 1}, np.exp(-np.array([[1e6, 1e6 + 0.j], [0, 1e6]])))

graph.sum_product(4, 1)
graph.beliefs()

print('n0', graph.node_belief['n0'])
print('n1', graph.node_belief['n1'])
print('n2', graph.node_belief['n2'])

