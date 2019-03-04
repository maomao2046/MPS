import numpy as np
import matplotlib.pyplot as plt
import matrix_product_state as mps
import copy as cp
import wavefunction as wf
import FactorGraph as fg


psi = np.array([[[[1., 1.], [0., 0.]], [[0.j, 0.], [0.j, 0.j]]], [[[1., 1.j], [0., 0.j]], [[0., 0.], [0., 0.]]]])
wf = wf.wavefunction()
wf.addwf(psi, 4, 2)

z = np.array([[1, 0], [0, -1]])

model = mps.MPS('OPEN')
model.wavefunction2mps(psi, 100)

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
graph.add_node(2, 'n10')
graph.add_node(8, 'n11')
graph.add_node(4, 'n12')
graph.add_node(4, 'n13')
graph.add_node(8, 'n14')
graph.add_node(2, 'n15')

graph.add_factor({'n0': 0, 'n4': 1}, model.mps[0])
graph.add_factor({'n4': 0, 'n5': 1}, model.mps[1].astype('complex128'))
graph.add_factor({'n5': 0, 'n1': 1, 'n6': 2}, model.mps[2])
graph.add_factor({'n6': 0, 'n7': 1}, model.mps[3].astype('complex128'))
graph.add_factor({'n7': 0, 'n2': 1, 'n8': 2}, model.mps[4])
graph.add_factor({'n8': 0, 'n9': 1}, model.mps[5].astype('complex128'))
graph.add_factor({'n9': 0, 'n3': 1}, model.mps[6])
graph.add_factor({'n0': 0, 'n10': 1}, np.conj(model.mps[0]))
graph.add_factor({'n10': 0, 'n11': 1}, np.conj(model.mps[1].astype('complex128')))
graph.add_factor({'n11': 0, 'n1': 1, 'n12': 2}, np.conj(model.mps[2]))
graph.add_factor({'n12': 0, 'n13': 1}, np.conj(model.mps[3].astype('complex128')))
graph.add_factor({'n13': 0, 'n2': 1, 'n14': 2}, np.conj(model.mps[4]))
graph.add_factor({'n14': 0, 'n15': 1}, np.conj(model.mps[5].astype('complex128')))
graph.add_factor({'n15': 0, 'n3': 1}, np.conj(model.mps[6]))
'''
graph.add_factor({'n0': 0, 'n4': 1}, model.mps[0] * np.conj(model.mps[0]))
graph.add_factor({'n4': 0, 'n5': 1}, model.mps[1].astype('complex128') * np.conj(model.mps[1].astype('complex128')))
graph.add_factor({'n5': 0, 'n1': 1, 'n6': 2}, model.mps[2] * np.conj(model.mps[2]))
graph.add_factor({'n6': 0, 'n7': 1}, model.mps[3].astype('complex128') * np.conj(model.mps[3].astype('complex128')))
graph.add_factor({'n7': 0, 'n2': 1, 'n8': 2}, model.mps[4] * np.conj(model.mps[4]))
graph.add_factor({'n8': 0, 'n9': 1}, model.mps[5].astype('complex128') * np.conj(model.mps[5].astype('complex128')))
graph.add_factor({'n9': 0, 'n3': 1}, model.mps[6] * np.conj(model.mps[6]))
'''
n0_belief0 = []
n0_belief1 = []
for i in range(1, 29):
    graph.sum_product(i, 1e-6)
    graph.beliefs()
    n0_belief0.append(graph.node_belief['n0'][0])
    n0_belief1.append(graph.node_belief['n0'][1])
    print(i)

for i in range(4):
    print('n' + str(i), graph.node_belief['n' + str(i)])

plt.figure()
plt.plot(range(len(n0_belief0)), np.real(n0_belief0), 'o')
plt.plot(range(len(n0_belief1)), np.real(n0_belief1), 'o')
plt.legend(['P(n0=up)', 'P(n0=down)'])
plt.xlabel('BP iterations')
plt.ylabel('Probability')
plt.show()









