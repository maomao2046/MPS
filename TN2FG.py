import numpy as np
import matplotlib.pyplot as plt
import matrix_product_state as mps
import copy as cp
import wavefunction as wf
import FactorGraph as fg


psi = np.array([[[[1., 0.], [0., 0.]], [[0., 0.], [-1.j, 2.]]], [[[4., 1.j], [2., 0.j]], [[3.j, 0.], [0., 0.j]]]])
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

graph = fg.Graph()

'''
# mps to FG (density matrix, with loops)
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
graph.add_factor({'n0': 0, 'n4': 1}, model.mps[0].astype('complex128'))
graph.add_factor({'n4': 0, 'n5': 1}, model.mps[1].astype('complex128'))
graph.add_factor({'n5': 0, 'n1': 1, 'n6': 2}, model.mps[2].astype('complex128'))
graph.add_factor({'n6': 0, 'n7': 1}, model.mps[3].astype('complex128'))
graph.add_factor({'n7': 0, 'n2': 1, 'n8': 2}, model.mps[4].astype('complex128'))
graph.add_factor({'n8': 0, 'n9': 1}, model.mps[5].astype('complex128'))
graph.add_factor({'n9': 0, 'n3': 1}, model.mps[6].astype('complex128'))
graph.add_factor({'n0': 0, 'n10': 1}, np.conj(model.mps[0].astype('complex128')))
graph.add_factor({'n10': 0, 'n11': 1}, np.conj(model.mps[1].astype('complex128')))
graph.add_factor({'n11': 0, 'n1': 1, 'n12': 2}, np.conj(model.mps[2].astype('complex128')))
graph.add_factor({'n12': 0, 'n13': 1}, np.conj(model.mps[3].astype('complex128')))
graph.add_factor({'n13': 0, 'n2': 1, 'n14': 2}, np.conj(model.mps[4].astype('complex128')))
graph.add_factor({'n14': 0, 'n15': 1}, np.conj(model.mps[5].astype('complex128')))
graph.add_factor({'n15': 0, 'n3': 1}, np.conj(model.mps[6].astype('complex128')))
'''
'''
# mps to FG (Tree)
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
'''

# mps to FG (real factors)
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
graph.add_factor({'n0': 0, 'n4': 1}, model.mps[0] * np.conj(model.mps[0]))
graph.add_factor({'n4': 0, 'n5': 1}, model.mps[1].astype('complex128') * np.conj(model.mps[1].astype('complex128')))
graph.add_factor({'n5': 0, 'n1': 1, 'n6': 2}, model.mps[2] * np.conj(model.mps[2]))
graph.add_factor({'n6': 0, 'n7': 1}, model.mps[3].astype('complex128') * np.conj(model.mps[3].astype('complex128')))
graph.add_factor({'n7': 0, 'n2': 1, 'n8': 2}, model.mps[4] * np.conj(model.mps[4]))
graph.add_factor({'n8': 0, 'n9': 1}, model.mps[5].astype('complex128') * np.conj(model.mps[5].astype('complex128')))
graph.add_factor({'n9': 0, 'n3': 1}, model.mps[6] * np.conj(model.mps[6]))


n0_belief0 = []
n0_belief1 = []
n1_belief0 = []
n1_belief1 = []
n2_belief0 = []
n2_belief1 = []
n3_belief0 = []
n3_belief1 = []

denfg_beliefs = {}
for n in graph.nodes:
    denfg_beliefs[n] = []

for i in range(1, 20):
    graph.sum_product(i, 1e-6)
    graph.beliefs()
    graph.DENFG_beliefs()
    n0_belief0.append(graph.node_belief['n0'][0])
    n0_belief1.append(graph.node_belief['n0'][1])
    n1_belief0.append(graph.node_belief['n1'][0])
    n1_belief1.append(graph.node_belief['n1'][1])
    n2_belief0.append(graph.node_belief['n2'][0])
    n2_belief1.append(graph.node_belief['n2'][1])
    n3_belief0.append(graph.node_belief['n3'][0])
    n3_belief1.append(graph.node_belief['n3'][1])
    for n in graph.nodes:
        denfg_beliefs[n].append(graph.DENFGbeliefs[n])
    print(i)

for n in graph.nodes:
    denfg_beliefs[n] = np.array(denfg_beliefs[n])

for i in range(4):
    print('n' + str(i), graph.node_belief['n' + str(i)])

plt.figure()
plt.subplot()
plt.plot(range(len(n0_belief0)), np.real(n0_belief0), 'o', color='green')
plt.plot(range(len(n0_belief1)), np.real(n0_belief1), 'P', color='green')
plt.plot(range(len(denfg_beliefs['n0'][:, 0])), np.real(denfg_beliefs['n0'][:, 0]), 'o', color='blue')
plt.plot(range(len(denfg_beliefs['n0'][:, 1])), np.real(denfg_beliefs['n0'][:, 1]), 'P', color='blue')
plt.title('$n_0$ (factor graph) vs $S_0$ (mps) measurements')
plt.xlabel('# BP iterations')
plt.ylabel('Probability', color='green')
plt.tick_params(axis='y', labelcolor='tab:green')
plt.yticks(list(np.array(range(0, 11)) / 10.))
plt.ylim([0, 1])
plt.legend(['$P(n_0=+1)$', '$P(n_0=-1)$', '$P_{DENFG}(n_0=+1)$', '$P_{DENFG}(n_0=-1)$'], loc='upper left')
plt.twinx()
plt.plot(range(len(n0_belief1)), np.ones(len(n0_belief0)) * np.real(expectation_z0_wf[0]), color='red')
plt.plot(range(len(n0_belief1)), np.real(n0_belief0) - np.real(n0_belief1), 'v', color='red')
plt.ylabel('Magnetization', color='tab:red')
plt.tick_params(axis='y', labelcolor='tab:red')
plt.yticks(list(np.array(range(-10, 12, 2)) / 10.))
plt.ylim([-1, 1])
plt.legend(['$<\psi|\sigma_z(n_0)|\psi>$', '$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
plt.tight_layout()
plt.show()

plt.figure()
plt.subplot()
plt.plot(range(len(n1_belief0)), np.real(n1_belief0), 'o', color='green')
plt.plot(range(len(n1_belief1)), np.real(n1_belief1), 'P', color='green')
plt.plot(range(len(denfg_beliefs['n1'][:, 0])), np.real(denfg_beliefs['n1'][:, 0]), 'o', color='blue')
plt.plot(range(len(denfg_beliefs['n1'][:, 1])), np.real(denfg_beliefs['n1'][:, 1]), 'P', color='blue')
plt.title('$n_1$ (factor graph) vs $S_1$ (mps) measurements')
plt.xlabel('# BP iterations')
plt.ylabel('Probability', color='green')
plt.tick_params(axis='y', labelcolor='tab:green')
plt.yticks(list(np.array(range(0, 11)) / 10.))
plt.ylim([0, 1])
plt.legend(['$P(n_1=+1)$', '$P(n_1=-1)$'], loc='upper left')
plt.twinx()
plt.plot(range(len(n1_belief1)), np.ones(len(n1_belief0)) * np.real(expectation_z1_wf[0]), color='red')
plt.plot(range(len(n1_belief1)), np.real(n1_belief0) - np.real(n1_belief1), 'v', color='red')
plt.ylabel('Magnetization', color='tab:red')
plt.tick_params(axis='y', labelcolor='tab:red')
plt.yticks(list(np.array(range(-10, 12, 2)) / 10.))
plt.ylim([-1, 1])
plt.legend(['$<\psi|\sigma_z(n_1)|\psi>$', '$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
plt.tight_layout()
plt.show()

plt.figure()
plt.subplot()
plt.plot(range(len(n2_belief0)), np.real(n2_belief0), 'o', color='green')
plt.plot(range(len(n2_belief1)), np.real(n2_belief1), 'P', color='green')
plt.plot(range(len(denfg_beliefs['n2'][:, 0])), np.real(denfg_beliefs['n2'][:, 0]), 'o', color='blue')
plt.plot(range(len(denfg_beliefs['n2'][:, 1])), np.real(denfg_beliefs['n2'][:, 1]), 'P', color='blue')
plt.title('$n_2$ (factor graph) vs $S_2$ (mps) measurements')
plt.xlabel('# BP iterations')
plt.ylabel('Probability', color='green')
plt.tick_params(axis='y', labelcolor='tab:green')
plt.yticks(list(np.array(range(0, 11)) / 10.))
plt.ylim([0, 1])
plt.legend(['$P(n_2=+1)$', '$P(n_2=-1)$'], loc='upper left')
plt.twinx()
plt.plot(range(len(n2_belief1)), np.ones(len(n2_belief0)) * np.real(expectation_z2_wf[0]), color='red')
plt.plot(range(len(n2_belief1)), np.real(n2_belief0) - np.real(n2_belief1), 'v', color='red')
plt.ylabel('Magnetization', color='tab:red')
plt.tick_params(axis='y', labelcolor='tab:red')
plt.yticks(list(np.array(range(-10, 12, 2)) / 10.))
plt.ylim([-1, 1])
plt.legend(['$<\psi|\sigma_z(n_2)|\psi>$', '$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
plt.tight_layout()
plt.show()

plt.figure()
plt.subplot()
plt.plot(range(len(n3_belief0)), np.real(n3_belief0), 'o', color='green')
plt.plot(range(len(n3_belief1)), np.real(n3_belief1), 'P', color='green')
plt.plot(range(len(denfg_beliefs['n3'][:, 0])), np.real(denfg_beliefs['n3'][:, 0]), 'o', color='blue')
plt.plot(range(len(denfg_beliefs['n3'][:, 1])), np.real(denfg_beliefs['n3'][:, 1]), 'P', color='blue')
plt.title('$n_3$ (factor graph) vs $S_3$ (mps) measurements')
plt.xlabel('# BP iterations')
plt.ylabel('Probability', color='green')
plt.tick_params(axis='y', labelcolor='tab:green')
plt.yticks(list(np.array(range(0, 11)) / 10.))
plt.ylim([0, 1])
plt.legend(['$P(n_3=+1)$', '$P(n_3=-1)$'], loc='upper left')
plt.twinx()
plt.plot(range(len(n3_belief1)), np.ones(len(n3_belief0)) * np.real(expectation_z3_wf[0]), color='red')
plt.plot(range(len(n3_belief1)), np.real(n3_belief0) - np.real(n3_belief1), 'v', color='red')
plt.ylabel('Magnetization', color='tab:red')
plt.tick_params(axis='y', labelcolor='tab:red')
plt.yticks(list(np.array(range(-10, 12, 2)) / 10.))
plt.ylim([-1, 1])
plt.legend(['$<\psi|\sigma_z(n_3)|\psi>$', '$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
plt.tight_layout()
plt.show()



