import numpy as np
from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.pyplot as plt
import wavefunction as wf
import matrix_product_state as mps
import DEnFG as denfg
import time

# 1D AKLT
n = 6
t_max = 40
physical_spins = []
virtual_spins = []
for i in range(n):
    physical_spins.append('n' + str(i))
for i in range(2 * n):
    virtual_spins.append('n' + str(i + n))

spin1_tensor = np.array([[[1, 0], [0, 0]], [[0, 1/np.sqrt(2)], [1/np.sqrt(2), 0]], [[0, 0], [0, 1]]])
#spin1_tensor = np.array([[[0, np.sqrt(2. / 3)], [0, 0]], [[- np.sqrt(1. / 3), 0], [0, np.sqrt(1. / 3)]], [[0, 0], [- np.sqrt(2. / 3), 0]]])
singlet = np.array([[0, 1/np.sqrt(2)], [-1/np.sqrt(2), 0]])
aklt = mps.MPS('PBC')
for i in range(n):
    aklt.add_physical_tensor(spin1_tensor, 0)
    aklt.add_virtual_tensor(singlet)
z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
x = 1./np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
y = 1j/np.sqrt(2) * np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]])
operator = y

graph = denfg.Graph()
for i in range(n):
    graph.add_node(3, physical_spins[i])
for i in range(2 * n):
    graph.add_node(2, virtual_spins[i])

for i in range(n):
    graph.add_factor({virtual_spins[2 * i - 1]: 0, physical_spins[i]: 1, virtual_spins[2 * i]: 2}, aklt.mps[0].astype('complex128'))
    graph.add_factor({virtual_spins[2 * i]: 0, virtual_spins[2 * i + 1]: 1}, aklt.mps[1].astype('complex128'))
#graph.factors['f0'][1] += (np.random.rand(2, 3, 2) + np.random.rand(2, 3, 2) * 1j) * 1e-1
#graph.factors['f2'][1] += (np.random.rand(2, 3, 2) + np.random.rand(2, 3, 2) * 1j) * 1e-2
#graph.factors['f4'][1] += (np.random.rand(2, 3, 2) + np.random.rand(2, 3, 2) * 1j) * 1e-3
#graph.factors['f6'][1] += (np.random.rand(2, 3, 2) + np.random.rand(2, 3, 2) * 1j) * 1e-1



expectation_graph = np.zeros((n, t_max), dtype=complex)
expectation_mps = np.zeros((n, t_max), dtype=complex)

for t in range(t_max):
    graph.sum_product(t, 1e-6)
    graph.calc_node_belief()
    for l in range(n):
        expectation_graph[l, t] = np.trace(np.matmul(graph.node_belief[physical_spins[l]], operator))
        expectation_mps[l, t] = aklt.OneSpinMeasurement(operator, l)

leg = []
plt.figure()
for i in range(n):
    leg.append('$<\sigma_z(' + str(i) + ')>_{DE-NFG}$')
    leg.append('$<\sigma_z(' + str(i) + ')>_{MPS}$')
    plt.plot(range(t_max), expectation_graph[i, :], 'o')
    plt.plot(range(t_max), expectation_mps[i, :])

plt.xticks(range(0, t_max, 2))
plt.ylabel('$m_z$')
plt.xlabel('# of BP iterations')
#plt.legend(leg)
plt.grid()
plt.show()








