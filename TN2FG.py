import numpy as np
import matplotlib.pyplot as plt
import matrix_product_state as mps
import copy as cp
import wavefunction as wf
import FactorGraph as fg


psi = np.array([[[[0., 0.], [-1., 0.]], [[0., 0.], [0., 1.]]], [[[0., 0.], [0., 0.]], [[1.j, 0.], [0., 0.]]]])
wf = wf.wavefunction()
wf.addwf(psi, 4, 2)

z = np.array([[1, 0], [0, -1]])

model = mps.MPS('OPEN')
model.wavefunction2mps(psi, 10)

expectation_z0_mps = model.SingleSpinMeasure(0, z)
expectation_z1_mps = model.SingleSpinMeasure(1, z)
expectation_z2_mps = model.SingleSpinMeasure(2, z)
expectation_z3_mps = model.SingleSpinMeasure(3, z)

expectation_z0_wf = wf.SingleSpinMeasurement(0, z)
expectation_z1_wf = wf.SingleSpinMeasurement(1, z)
expectation_z2_wf = wf.SingleSpinMeasurement(2, z)
expectation_z3_wf = wf.SingleSpinMeasurement(3, z)

graph = fg.Graph(10)
graph.add_node(2, 'n0')
graph.add_node(2, 'n1')
graph.add_node(2, 'n2')
graph.add_node(2, 'n3')
graph.add_node(2, 'n4')
graph.add_node(8, 'n5')
graph.add_node(4, 'n6')
graph.add_node(4, 'n7')
graph.add_node(8, 'n8')
graph.add_node(2, 'n9')
graph.add_factor(['n0', 'n4'], model.mps[0])
graph.add_factor(['n4', 'n5'], model.mps[1].astype('complex128'))
graph.add_factor(['n5', 'n1', 'n6'], model.mps[2])
graph.add_factor(['n6', 'n7'], model.mps[3].astype('complex128'))
graph.add_factor(['n7', 'n2', 'n8'], model.mps[4])
graph.add_factor(['n8', 'n9'], model.mps[5].astype('complex128'))
graph.add_factor(['n9', 'n3'], model.mps[6])

graph.sum_product(10, 1e-6)


