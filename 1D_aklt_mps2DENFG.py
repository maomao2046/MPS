import numpy as np
import matplotlib.pyplot as plt
import matrix_product_state as mps
import copy as cp
import wavefunction as wf
import FactorGraph as fg

# 1D AKLT
n = 4
spin1_tensor = np.array([[[1, 0], [0, 0]], [[0, 1/np.sqrt(2)], [1/np.sqrt(2), 0]], [[0, 0], [0, 1]]])
#spin1_tensor = np.array([[[0, np.sqrt(2. / 3)], [0, 0]], [[- np.sqrt(1. / 3), 0], [0, np.sqrt(1. / 3)]], [[0, 0], [- np.sqrt(2. / 3), 0]]])
singlet = np.array([[0, 1/np.sqrt(2)], [-1/np.sqrt(2), 0]])
aklt = mps.MPS('PBC')
for i in range(n):
    aklt.add_physical_tensor(spin1_tensor, 0)
    aklt.add_virtual_tensor(singlet)
z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
x = 1/np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
y = 1j/np.sqrt(2) * np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]])

expectation_z0_mps = aklt.SingleSpinMeasure(z, 0)
expectation_z1_mps = aklt.SingleSpinMeasure(z, 1)
expectation_z2_mps = aklt.SingleSpinMeasure(z, 2)
expectation_z3_mps = aklt.SingleSpinMeasure(z, 3)

expectation_x0_mps = aklt.SingleSpinMeasure(x, 0)
expectation_x1_mps = aklt.SingleSpinMeasure(x, 1)
expectation_x2_mps = aklt.SingleSpinMeasure(x, 2)
expectation_x3_mps = aklt.SingleSpinMeasure(x, 3)

psi = aklt.mps2tensor()
wf = wf.wavefunction()
wf.addwf(psi, n, 3)

expectation_x0_wf = wf.SingleSpinMeasurement(0, x)
expectation_x1_wf = wf.SingleSpinMeasurement(1, x)
expectation_x2_wf = wf.SingleSpinMeasurement(2, x)
expectation_x3_wf = wf.SingleSpinMeasurement(3, x)
#expectation_x4_wf = wf.SingleSpinMeasurement(4, x)
#expectation_x5_wf = wf.SingleSpinMeasurement(5, x)


expectation_z0_wf = wf.SingleSpinMeasurement(0, z)
expectation_z1_wf = wf.SingleSpinMeasurement(1, z)
expectation_z2_wf = wf.SingleSpinMeasurement(2, z)
expectation_z3_wf = wf.SingleSpinMeasurement(3, z)
#expectation_z4_wf = wf.SingleSpinMeasurement(4, z)
#expectation_z5_wf = wf.SingleSpinMeasurement(5, z)


graph = fg.Graph()

graph.add_node(3, 'n0')
graph.add_node(3, 'n1')
graph.add_node(3, 'n2')
graph.add_node(3, 'n3')
graph.add_node(2, 'n4')
graph.add_node(2, 'n5')
graph.add_node(2, 'n6')
graph.add_node(2, 'n7')
graph.add_node(2, 'n8')
graph.add_node(2, 'n9')
graph.add_node(2, 'n10')
graph.add_node(2, 'n11')
graph.add_factor({'n11': 0, 'n0': 1, 'n4': 2}, aklt.mps[0].astype('complex128'))
graph.add_factor({'n4': 0, 'n5': 1}, aklt.mps[1].astype('complex128'))
graph.add_factor({'n5': 0, 'n1': 1, 'n6': 2}, aklt.mps[2].astype('complex128'))
graph.add_factor({'n6': 0, 'n7': 1}, aklt.mps[3].astype('complex128'))
graph.add_factor({'n7': 0, 'n2': 1, 'n8': 2}, aklt.mps[4].astype('complex128'))
graph.add_factor({'n8': 0, 'n9': 1}, aklt.mps[5].astype('complex128'))
graph.add_factor({'n9': 0, 'n3': 1, 'n10': 2}, aklt.mps[6].astype('complex128'))
graph.add_factor({'n10': 0, 'n11': 1}, aklt.mps[7].astype('complex128'))


n0_belief0 = []
n0_belief1 = []
n0_belief2 = []

n1_belief0 = []
n1_belief1 = []
n1_belief2 = []

n2_belief0 = []
n2_belief1 = []
n2_belief2 = []

n3_belief0 = []
n3_belief1 = []
n3_belief2 = []

denfg_beliefs = {}
for n in graph.nodes:
    denfg_beliefs[n] = []

for i in range(1, 20):
    graph.sum_product(i, 1e-6)
    graph.beliefs()
    graph.DENFG_beliefs()
    n0_belief0.append(graph.node_belief['n0'][0])
    n0_belief1.append(graph.node_belief['n0'][1])
    n0_belief2.append(graph.node_belief['n0'][2])

    n1_belief0.append(graph.node_belief['n1'][0])
    n1_belief1.append(graph.node_belief['n1'][1])
    n1_belief2.append(graph.node_belief['n1'][2])

    n2_belief0.append(graph.node_belief['n2'][0])
    n2_belief1.append(graph.node_belief['n2'][1])
    n2_belief2.append(graph.node_belief['n2'][2])

    n3_belief0.append(graph.node_belief['n3'][0])
    n3_belief1.append(graph.node_belief['n3'][1])
    n3_belief2.append(graph.node_belief['n3'][2])
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
plt.plot(range(len(n0_belief2)), np.real(n0_belief2), 'v', color='green')
#plt.plot(range(len(denfg_beliefs['n0'][:, 0])), np.real(denfg_beliefs['n0'][:, 0]), 'o', color='blue')
#plt.plot(range(len(denfg_beliefs['n0'][:, 1])), np.real(denfg_beliefs['n0'][:, 1]), 'P', color='blue')
plt.title('$n_0$ (factor graph) vs $S_0$ (mps) measurements')
plt.xlabel('# BP iterations')
plt.ylabel('Probability', color='green')
plt.tick_params(axis='y', labelcolor='tab:green')
plt.yticks(list(np.array(range(0, 11)) / 10.))
plt.ylim([0, 1])
#plt.legend(['$P(n_0=+1)$', '$P(n_0=-1)$', '$P_{DENFG}(n_0=+1)$', '$P_{DENFG}(n_0=-1)$'], loc='upper left')
plt.legend(['$P(n_0=+1)$', '$P(n_0=0)$', '$P(n_0=-1)$'], loc='upper left')
#plt.twinx()
#plt.plot(range(len(n0_belief1)), np.ones(len(n0_belief0)) * np.real(expectation_z0_wf[0]), color='red')
#plt.plot(range(len(n0_belief1)), np.real(n0_belief0) - np.real(n0_belief2), 'v', color='red')
#plt.ylabel('Magnetization', color='tab:red')
#plt.tick_params(axis='y', labelcolor='tab:red')
#plt.yticks(list(np.array(range(-10, 12, 2)) / 10.))
#plt.ylim([-1, 1])
#plt.legend(['$<\psi|\sigma_z(n_0)|\psi>$', '$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
#plt.legend(['$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
#plt.tight_layout()
plt.show()

plt.figure()
plt.subplot()
plt.plot(range(len(n1_belief0)), np.real(n1_belief0), 'o', color='green')
plt.plot(range(len(n1_belief1)), np.real(n1_belief1), 'P', color='green')
plt.plot(range(len(n1_belief2)), np.real(n1_belief2), 'v', color='green')

#plt.plot(range(len(denfg_beliefs['n1'][:, 0])), np.real(denfg_beliefs['n1'][:, 0]), 'o', color='blue')
#plt.plot(range(len(denfg_beliefs['n1'][:, 1])), np.real(denfg_beliefs['n1'][:, 1]), 'P', color='blue')
plt.title('$n_1$ (factor graph) vs $S_1$ (mps) measurements')
plt.xlabel('# BP iterations')
plt.ylabel('Probability', color='green')
plt.tick_params(axis='y', labelcolor='tab:green')
plt.yticks(list(np.array(range(0, 11)) / 10.))
plt.ylim([0, 1])
plt.legend(['$P(n_1=+1)$', '$P(n_1=0)$', '$P(n_1=-1)$'], loc='upper left')
#plt.twinx()
#plt.plot(range(len(n1_belief1)), np.ones(len(n1_belief0)) * np.real(expectation_z1_wf[0]), color='red')
#plt.plot(range(len(n1_belief1)), np.real(n1_belief0) - np.real(n1_belief1), 'v', color='red')
#plt.ylabel('Magnetization', color='tab:red')
#plt.tick_params(axis='y', labelcolor='tab:red')
#plt.yticks(list(np.array(range(-10, 12, 2)) / 10.))
#plt.ylim([-1, 1])
#plt.legend(['$<\psi|\sigma_z(n_1)|\psi>$', '$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
#plt.legend(['$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
#plt.tight_layout()
plt.show()

plt.figure()
plt.subplot()
plt.plot(range(len(n2_belief0)), np.real(n2_belief0), 'o', color='green')
plt.plot(range(len(n2_belief1)), np.real(n2_belief1), 'P', color='green')
plt.plot(range(len(n2_belief2)), np.real(n2_belief2), 'v', color='green')

#plt.plot(range(len(denfg_beliefs['n2'][:, 0])), np.real(denfg_beliefs['n2'][:, 0]), 'o', color='blue')
#plt.plot(range(len(denfg_beliefs['n2'][:, 1])), np.real(denfg_beliefs['n2'][:, 1]), 'P', color='blue')
plt.title('$n_2$ (factor graph) vs $S_2$ (mps) measurements')
plt.xlabel('# BP iterations')
plt.ylabel('Probability', color='green')
plt.tick_params(axis='y', labelcolor='tab:green')
plt.yticks(list(np.array(range(0, 11)) / 10.))
plt.ylim([0, 1])
plt.legend(['$P(n_2=+1)$', '$P(n_2=0)$', '$P(n_2=-1)$'], loc='upper left')
#plt.twinx()
#plt.plot(range(len(n2_belief1)), np.ones(len(n2_belief0)) * np.real(expectation_z2_wf[0]), color='red')
#plt.plot(range(len(n2_belief1)), np.real(n2_belief0) - np.real(n2_belief1), 'v', color='red')
#plt.ylabel('Magnetization', color='tab:red')
#plt.tick_params(axis='y', labelcolor='tab:red')
#plt.yticks(list(np.array(range(-10, 12, 2)) / 10.))
#plt.ylim([-1, 1])
#plt.legend(['$<\psi|\sigma_z(n_2)|\psi>$', '$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
#plt.legend(['$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
#plt.tight_layout()
plt.show()

plt.figure()
plt.subplot()
plt.plot(range(len(n3_belief0)), np.real(n3_belief0), 'o', color='green')
plt.plot(range(len(n3_belief1)), np.real(n3_belief1), 'P', color='green')
plt.plot(range(len(n3_belief2)), np.real(n3_belief2), 'v', color='green')

#plt.plot(range(len(denfg_beliefs['n3'][:, 0])), np.real(denfg_beliefs['n3'][:, 0]), 'o', color='blue')
#plt.plot(range(len(denfg_beliefs['n3'][:, 1])), np.real(denfg_beliefs['n3'][:, 1]), 'P', color='blue')
plt.title('$n_3$ (factor graph) vs $S_3$ (mps) measurements')
plt.xlabel('# BP iterations')
plt.ylabel('Probability', color='green')
plt.tick_params(axis='y', labelcolor='tab:green')
plt.yticks(list(np.array(range(0, 11)) / 10.))
plt.ylim([0, 1])
plt.legend(['$P(n_3=+1)$', '$P(n_3=0)$', '$P(n_3=-1)$'], loc='upper left')
#plt.twinx()
#plt.plot(range(len(n3_belief1)), np.ones(len(n3_belief0)) * np.real(expectation_z3_wf[0]), color='red')
#plt.plot(range(len(n3_belief1)), np.real(n3_belief0) - np.real(n3_belief1), 'v', color='red')
#plt.ylabel('Magnetization', color='tab:red')
#plt.tick_params(axis='y', labelcolor='tab:red')
#plt.yticks(list(np.array(range(-10, 12, 2)) / 10.))
#plt.ylim([-1, 1])
#plt.legend(['$<\psi|\sigma_z(n_3)|\psi>$', '$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
#plt.legend(['$1 \cdot P(+1) -1 \cdot P(-1)$'], loc='upper right')
#plt.tight_layout()
plt.show()
