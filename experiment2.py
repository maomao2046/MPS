import numpy as np
import matplotlib.pyplot as plt
import wavefunction as wf
import matrix_product_state as mps
import DEnFG as denfg

k = 6
n = 4
alphabet = 2
t_max = 20
error = 1e-6
psi = np.array([[[[1., 1.], [2., 0.5 + 3.j]], [[4., 8.j], [-1.j, 2.]]], [[[4., 1.j], [2., 1.j]], [[3.j, 4.], [2., 1.j]]]])
#psi = np.array([[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[1.j, 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]])

wf = wf.wavefunction()
wf.addwf(psi, n, alphabet)

z = np.array([[1, 0], [0, -1]])

expectation_z0_wf = wf.SingleSpinMeasurement(0, z)
expectation_z1_wf = wf.SingleSpinMeasurement(1, z)
expectation_z2_wf = wf.SingleSpinMeasurement(2, z)
expectation_z3_wf = wf.SingleSpinMeasurement(3, z)

expectation_z0_mps = []
expectation_z1_mps = []
expectation_z2_mps = []
expectation_z3_mps = []

expectation_z0_graph = []
expectation_z1_graph = []
expectation_z2_graph = []
expectation_z3_graph = []



for i in range(1, k):
    model = mps.MPS('OPEN')
    model.wavefunction2mps(psi, i)

    expectation_z0_mps.append(model.SingleSpinMeasure(z, 0))
    expectation_z1_mps.append(model.SingleSpinMeasure(z, 1))
    expectation_z2_mps.append(model.SingleSpinMeasure(z, 2))
    expectation_z3_mps.append(model.SingleSpinMeasure(z, 3))

    graph = denfg.Graph()
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
    graph.add_factor({'n0': 0, 'n4': 1}, model.mps[0].astype('complex128'))
    graph.add_factor({'n4': 0, 'n5': 1}, model.mps[1].astype('complex128'))
    graph.add_factor({'n5': 0, 'n1': 1, 'n6': 2}, model.mps[2].astype('complex128'))
    graph.add_factor({'n6': 0, 'n7': 1}, model.mps[3].astype('complex128'))
    graph.add_factor({'n7': 0, 'n2': 1, 'n8': 2}, model.mps[4].astype('complex128'))
    graph.add_factor({'n8': 0, 'n9': 1}, model.mps[5].astype('complex128'))
    graph.add_factor({'n9': 0, 'n3': 1}, model.mps[6].astype('complex128'))

    graph.sum_product(t_max, error)
    graph.calc_node_belief()

    expectation_z0_graph.append(np.trace(np.matmul(graph.node_belief['n0'], z)))
    expectation_z1_graph.append(np.trace(np.matmul(graph.node_belief['n1'], z)))
    expectation_z2_graph.append(np.trace(np.matmul(graph.node_belief['n2'], z)))
    expectation_z3_graph.append(np.trace(np.matmul(graph.node_belief['n3'], z)))


plt.figure()
plt.title('4 spins 1/2 wavefunction expectations: \n MPS vs DE-NFG (after convergence)')
plt.plot(range(1, k), expectation_z0_graph)
plt.plot(range(1, k), np.ones(k - 1) * expectation_z0_wf[0])
plt.plot(range(1, k), expectation_z0_mps)
plt.ylabel('$<\psi|\sigma_z(n_0)|\psi>$')
plt.xlabel('D - bond dimension kept')
plt.ylim([0, 1])
plt.legend(['DE-NFG', 'true value', 'MPS'])
plt.show()


