from scipy.stats import unitary_group
import numpy as np
import DEnFG as denfg

g = denfg.Graph()

g.add_node(2, 'n0')
g.add_node(2, 'n1')
#g.add_node(3, 'n2')
#g.add_node(2, 'n3')

g.add_factor({'n0': 0, 'n1': 1}, np.random.rand(2, 2) + np.random.rand(2, 2) * 1j)
#g.add_factor({'n1': 0, 'n2': 1}, np.random.rand(2, 3) + np.random.rand(2, 3) * 1j)
#g.add_factor({'n2': 0, 'n3': 1}, np.random.rand(3, 2) + np.random.rand(3, 2) * 1j)

g.sum_product(10)
g.calc_node_partition()
g.calc_factor_partition()
bethe = g.bethe_partition()
z = g.calc_partition()

