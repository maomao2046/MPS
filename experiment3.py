import numpy as np
import matplotlib.pyplot as plt
import wavefunction as wf
import matrix_product_state as mps
import DEnFG as denfg

k = 2
n = 6
spins = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5']
alphabet = 2
t_max = 20
error = 1e-6

psi = np.array([[[[[[1., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]], [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                 [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]], [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                 [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., -1.j]]]]]])
'''
psi = np.array([[[[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[-1.j, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                [[[[[1.j, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                 [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]]])
'''
wf = wf.wavefunction()
wf.addwf(psi, n, alphabet)
#wf.random_wavefunction(n, alphabet)

z = np.array([[1, 0], [0, -1]])

model = mps.MPS('OPEN')
model.wavefunction2mps(wf.tensor, k)

expectation_mps = []
expectation_wf = []
for i in range(n):
    expectation_mps.append(model.SingleSpinMeasure(z, i))
    expectation_wf.append(wf.SingleSpinMeasurement(i, z))

expectation_graph = np.zeros((n, t_max), dtype=complex)

graph = denfg.Graph()
graph.add_node(alphabet, 'n0')
graph.add_node(alphabet, 'n1')
graph.add_node(alphabet, 'n2')
graph.add_node(alphabet, 'n3')
graph.add_node(alphabet, 'n4')
graph.add_node(alphabet, 'n5')
graph.add_node(2, 'n6')
graph.add_node(32, 'n7')
graph.add_node(4, 'n8')
graph.add_node(16, 'n9')
graph.add_node(8, 'n10')
graph.add_node(8, 'n11')
graph.add_node(16, 'n12')
graph.add_node(4, 'n13')
graph.add_node(32, 'n14')
graph.add_node(2, 'n15')


graph.add_factor({'n0': 0, 'n6': 1}, model.mps[0].astype('complex128'))
graph.add_factor({'n6': 0, 'n7': 1}, model.mps[1].astype('complex128'))
graph.add_factor({'n7': 0, 'n1': 1, 'n8': 2}, model.mps[2].astype('complex128'))
graph.add_factor({'n8': 0, 'n9': 1}, model.mps[3].astype('complex128'))
graph.add_factor({'n9': 0, 'n2': 1, 'n10': 2}, model.mps[4].astype('complex128'))
graph.add_factor({'n10': 0, 'n11': 1}, model.mps[5].astype('complex128'))
graph.add_factor({'n11': 0, 'n3': 1, 'n12': 2}, model.mps[6].astype('complex128'))
graph.add_factor({'n12': 0, 'n13': 1}, model.mps[7].astype('complex128'))
graph.add_factor({'n13': 0, 'n4': 1, 'n14': 2}, model.mps[8].astype('complex128'))
graph.add_factor({'n14': 0, 'n15': 1}, model.mps[9].astype('complex128'))
graph.add_factor({'n15': 0, 'n5': 1}, model.mps[10].astype('complex128'))

for t in range(t_max):
    graph.sum_product(t, error)
    graph.calc_node_belief()
    for i in range(n):
        expectation_graph[i, t] = np.trace(np.matmul(graph.node_belief[spins[i]], z))

'''
plt.figure()
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.title('$<\psi|\sigma_z(' + spins[i] + ')|\psi>$  \n MPS (D = 3) vs DE-NFG')
    plt.plot(range(t_max), expectation_graph[i, :], 'o')
    plt.plot(range(t_max), np.ones(t_max) * expectation_wf[i][0])
    plt.plot(range(t_max), np.ones(t_max) * expectation_mps[i])
    plt.ylim([-1.1, 1.1])
    plt.ylabel('$<\psi|\sigma_z(n_0)|\psi>$')
    plt.xlabel('# BP iterations')
    plt.legend(['DE-NFG', 'true value', 'MPS'])
plt.show()
'''

for i in range(n):
    plt.figure()
    plt.title('$<\psi|\sigma_z(' + spins[i] + ')|\psi>$  \n MPS (D = 3) vs DE-NFG')
    plt.plot(range(t_max), expectation_graph[i, :], 'o')
    plt.plot(range(t_max), np.ones(t_max) * expectation_wf[i][0])
    plt.plot(range(t_max), np.ones(t_max) * expectation_mps[i])
    plt.ylim([-1.1, 1.1])
    plt.ylabel('$<\psi|\sigma_z(n_0)|\psi>$')
    plt.xlabel('# BP iterations')
    plt.legend(['DE-NFG', 'true value', 'MPS'])
    plt.show()