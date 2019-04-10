import numpy as np
from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.pyplot as plt
import wavefunction as wf
import matrix_product_state as mps
import DEnFG as denfg
import time



k = 2
n = 10
spins = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
alphabet = 2
t_max = 25
error = 1e-6
'''
psi = np.array([[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]], [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                 [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]], [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                 [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]])

psi = np.array([[[[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[0.j, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                [[[[[0.j, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                 [[[[0., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.]]]]]])



psi = np.array([[[[[[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                 [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                [[[[[0.j, 0.], [1.j, 0.j]], [[0., 0.j], [-1.j, 0.]]],
                [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                 [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]]],
                [[[[[[0., 0.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                   [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                  [[[[0., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                   [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                 [[[[[0.j, 0.], [0.j, 0.j]], [[0., 0.j], [-1.j, 0.]]],
                   [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                  [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                   [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]]],
                [[[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                    [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                   [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                    [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                  [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                    [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                   [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                    [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]],
                 [[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                    [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                   [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                    [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                  [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                    [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                   [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                    [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]]]],
                [[[[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                     [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                    [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                     [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                   [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                     [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                    [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                     [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]],
                  [[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                     [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                    [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                     [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                   [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                     [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                    [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                     [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]]],
                 [[[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                     [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                    [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                     [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                   [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                     [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                    [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                     [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]],
                  [[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                     [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                    [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                     [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                   [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                     [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                    [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                     [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]]]]],
                [[[[[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                    [[[[[0.j, 0.], [1.j, 0.j]], [[0., 0.j], [-1.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]]],
                   [[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                    [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]]],
                  [[[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                    [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]],
                   [[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                    [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]]]],
                 [[[[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                    [[[[[0.j, 0.], [1.j, 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]],
                   [[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                    [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]]]],
                  [[[[[[[0., 1.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                    [[[[[0.j, 0.], [1.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]]],
                   [[[[[[0., 1.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., -1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                    [[[[[0.j, 0.], [0.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                      [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                     [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                      [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]]]]]]])

shape = np.ones(n, dtype=int) * alphabet
psi = np.zeros(shape, dtype=complex)
psi[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1.
psi[0, 1, 0, 0, 0, 0, 0, 0, 0, 0] = 1.j
psi[0, 0, 1, 0, 0, 0, 0, 0, 0, 0] = -1.
psi[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] = -1.
psi[0, 0, 0, 0, 1, 0, 0, 0, 0, 0] = 2.
psi[0, 0, 0, 0, 0, 1, 0, 0, 0, 0] = 0.3
psi[0, 0, 0, 0, 0, 0, 1, 0, 0, 0] = 1.
psi[0, 0, 0, 0, 0, 0, 0, 1, 0, 0] = 1.
psi[0, 0, 0, 0, 0, 0, 0, 0, 1, 0] = 0.6j
psi[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] = 1.
'''
shape = np.ones(n, dtype=int) * alphabet
psi = np.zeros(shape, dtype=complex)
psi[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1.
psi[0, 1, 0, 0, 0, 0, 0, 0, 0, 0] = 1.j
psi[0, 0, 1, 0, 0, 0, 0, 0, 0, 0] = -1.
psi[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] = -1.
psi[0, 0, 0, 0, 1, 0, 0, 0, 0, 0] = -1.
psi[0, 0, 0, 0, 0, 1, 0, 0, 0, 0] = 1.
psi[0, 0, 0, 0, 0, 0, 1, 0, 0, 0] = 1.
psi[0, 0, 0, 0, 0, 0, 0, 1, 0, 0] = 1.
psi[0, 0, 0, 0, 0, 0, 0, 0, 1, 0] = 1.j
psi[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] = 1.

wf = wf.wavefunction()
wf.addwf(psi, n, alphabet)
#wf.random_wavefunction(n, alphabet)

z = np.array([[1, 0], [0, -1]])
x = np.array([[0, 1], [1, 0]])



expectation_wf = []
for i in range(n):
    expectation_wf.append(wf.SingleSpinMeasurement(i, z))

expectation_mps = np.zeros((n, t_max), dtype=complex)
expectation_graph = np.zeros((n, t_max), dtype=complex)
correlations_mps = np.zeros((n, t_max), dtype=complex)
correlations_graph = np.zeros((n, t_max), dtype=complex)


model = mps.MPS('OPEN')
s = time.time()
model.wavefunction2mps(wf.tensor, k)
e = time.time()
print('wf 2 mps time is:', e - s)

for t in range(t_max):
    s = time.time()
    graph = denfg.Graph()
    graph.add_node(alphabet, 'n0')
    graph.add_node(alphabet, 'n1')
    graph.add_node(alphabet, 'n2')
    graph.add_node(alphabet, 'n3')
    graph.add_node(alphabet, 'n4')
    graph.add_node(alphabet, 'n5')
    graph.add_node(alphabet, 'n6')
    graph.add_node(alphabet, 'n7')
    graph.add_node(alphabet, 'n8')
    graph.add_node(alphabet, 'n9')

    graph.add_node(2, 'n10')
    graph.add_node(512, 'n11')
    graph.add_node(4, 'n12')
    graph.add_node(256, 'n13')
    graph.add_node(8, 'n14')
    graph.add_node(128, 'n15')
    graph.add_node(16, 'n16')
    graph.add_node(64, 'n17')
    graph.add_node(32, 'n18')
    graph.add_node(32, 'n19')
    graph.add_node(64, 'n20')
    graph.add_node(16, 'n21')
    graph.add_node(128, 'n22')
    graph.add_node(8, 'n23')
    graph.add_node(256, 'n24')
    graph.add_node(4, 'n25')
    graph.add_node(512, 'n26')
    graph.add_node(2, 'n27')

    graph.add_factor({'n0': 0, 'n10': 1}, model.mps[0].astype('complex128'))
    graph.add_factor({'n10': 0, 'n11': 1}, model.mps[1].astype('complex128'))
    graph.add_factor({'n11': 0, 'n1': 1, 'n12': 2}, model.mps[2].astype('complex128'))
    graph.add_factor({'n12': 0, 'n13': 1}, model.mps[3].astype('complex128'))
    graph.add_factor({'n13': 0, 'n2': 1, 'n14': 2}, model.mps[4].astype('complex128'))
    graph.add_factor({'n14': 0, 'n15': 1}, model.mps[5].astype('complex128'))
    graph.add_factor({'n15': 0, 'n3': 1, 'n16': 2}, model.mps[6].astype('complex128'))
    graph.add_factor({'n16': 0, 'n17': 1}, model.mps[7].astype('complex128'))
    graph.add_factor({'n17': 0, 'n4': 1, 'n18': 2}, model.mps[8].astype('complex128'))
    graph.add_factor({'n18': 0, 'n19': 1}, model.mps[9].astype('complex128'))
    graph.add_factor({'n19': 0, 'n5': 1, 'n20': 2}, model.mps[10].astype('complex128'))
    graph.add_factor({'n20': 0, 'n21': 1}, model.mps[11].astype('complex128'))
    graph.add_factor({'n21': 0, 'n6': 1, 'n22': 2}, model.mps[12].astype('complex128'))
    graph.add_factor({'n22': 0, 'n23': 1}, model.mps[13].astype('complex128'))
    graph.add_factor({'n23': 0, 'n7': 1, 'n24': 2}, model.mps[14].astype('complex128'))
    graph.add_factor({'n24': 0, 'n25': 1}, model.mps[15].astype('complex128'))
    graph.add_factor({'n25': 0, 'n8': 1, 'n26': 2}, model.mps[16].astype('complex128'))
    graph.add_factor({'n26': 0, 'n27': 1}, model.mps[17].astype('complex128'))
    graph.add_factor({'n27': 0, 'n9': 1}, model.mps[18].astype('complex128'))
    e = time.time()
    print('generating the graph time:', e - s)

    s = time.time()
    graph.sum_product(t, error)
    e = time.time()
    print('\n\n')
    print(str(t) + ' iterations of sum product finished at time:', e - s)
    s = time.time()
    graph.calc_node_belief()
    e = time.time()
    print('belief calc finished at time:', e - s)
    for l in range(n):
        s = time.time()
        expectation_mps[l, t] = model.SingleSpinMeasure(z, l)
        e = time.time()
        print('mps expectation finished in time: ',e - s)
        s = time.time()
        expectation_graph[l, t] = np.trace(np.matmul(graph.node_belief[spins[l]], z))
        e = time.time()
        print('graph expectation finished in time: ', e - s)
        #correlations_mps[l, i, t] = model.TwoSpinsMeasurement(0, l, z, z)
        s = time.time()
        correlations_graph[l, t] = np.trace(np.matmul(graph.node_belief[spins[0]], z)) * np.trace(np.matmul(graph.node_belief[spins[l]], z))
        e = time.time()
        print('graph correlations finished in time: ', e - s)


for i in range(n):
    '''
    plt.figure()
    plt.subplot(211)
    plt.title('$<\psi|\sigma_z(' + spins[i] + ')|\psi>$ MPS vs DE-NFG \n expectation vs MPS bond dim')
    plt.plot(range(k), expectation_graph[i, :, t_max - 1], 'o')
    plt.plot(range(k), np.ones(k) * expectation_wf[i][0])
    plt.plot(range(k), expectation_mps[i, :, t_max - 1])
    plt.ylim([-1.1, 1.1])
    plt.ylabel('$<\sigma_z>$')
    plt.xlabel('D - bond dimension')
    plt.legend(['DE-NFG', 'true value', 'MPS'])
    
    plt.figure()
    #plt.subplot(212)
    plt.title('expectation vs DE-NFG BP iterations')
    plt.plot(range(t_max), expectation_graph[i, :], 'o')
    plt.plot(range(t_max), np.ones(t_max) * expectation_wf[i][0])
    plt.plot(range(t_max), expectation_mps[i, :])
    plt.ylim([-1.1, 1.1])
    plt.ylabel('$<\sigma_z>$')
    plt.xlabel('# BP iterations')
    plt.legend(['DE-NFG', 'true value', 'MPS'])
    plt.show()
    
    nrows, ncols = np.real(expectation_graph[0, :, :]).shape
    kk = np.linspace(0, k, ncols)
    tt = np.linspace(0, t_max, nrows)
    K, T = np.meshgrid(kk, tt)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.set_title(spins[i])
    ax.plot_surface(K, T, np.real(expectation_graph[i, :, :]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(20, 30)
    ax.set_xlabel('bond dimension')
    ax.set_ylabel('# bp iterations')
    ax.set_zlabel('$<\sigma_z>$')
    ax.set_zlim([-1.1, 1.1])
    plt.show()
    '''
    # plt.subplot(212)
    plt.figure()
    plt.plot(range(t_max), expectation_mps[i, :], linewidth=4)
    plt.plot(range(t_max), expectation_graph[i, :], 'o', ms=20)
    # plt.plot(range(t_max), np.ones(t_max) * expectation_wf[i][0], linewidth=2)
    plt.ylim([-1.1, 1.1])
    plt.xticks(range(t_max))
    #plt.ylabel('$<\psi|\sigma_{z_{' + str(i + 1) + '}}|\psi>$', fontsize=18)
    #plt.xlabel('# of BP iterations', fontsize=18)
    # plt.legend(['$<\sigma_z>_{DE-NFG}$', '$<\sigma_z>_{exact}$', '$<\sigma_z>_{MPS}$'])
    plt.legend(['$<\sigma_z>_{MPS}$', '$<\sigma_z>_{DE-NFG}$'], fontsize=12)
    plt.grid()
    plt.show()

plt.figure()
for i in range(t_max):
    plt.plot(range(n), correlations_graph[:, i])
#plt.plot(range(n), correlations_mps[:, bond, bp_iter])
plt.ylabel('$<z_0x_r>$')
plt.xlabel('r')
#plt.show()


