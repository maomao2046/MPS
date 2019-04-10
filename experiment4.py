import numpy as np
from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.pyplot as plt
import wavefunction as wf
import matrix_product_state as mps
import DEnFG as denfg


k = 4
n = 6
spins = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5']
alphabet = 2
t_max = 20
error = 1e-6
'''
psi = np.array([[[[[[0., 0.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                 [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                [[[[[0.j, 0.], [0.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                 [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]])
'''
psi = np.array([[[[[[-0.5, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1.j, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                [[[[[0.2j, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [-0.5, 0.j]], [[0.j, 0.], [0., 0.j]]]],
                 [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.]]]]]])


wf = wf.wavefunction()
wf.addwf(psi, n, alphabet)
#wf.random_wavefunction(n, alphabet)

z = np.array([[1, 0], [0, -1]])


expectation_wf = []
for i in range(n):
    expectation_wf.append(wf.SingleSpinMeasurement(i, z))

expectation_mps = np.zeros((n, k, t_max), dtype=complex)
expectation_graph = np.zeros((n, k, t_max), dtype=complex)


for i in range(1, k):
    model = mps.MPS('OPEN')
    model.wavefunction2mps(wf.tensor, i)
    for t in range(t_max):
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

        graph.sum_product(t, error)
        graph.calc_node_belief()
        print(i, t)
        for l in range(n):
            expectation_mps[l, i, t] = model.SingleSpinMeasure(z, l)
            expectation_graph[l, i, t] = np.trace(np.matmul(graph.node_belief[spins[l]], z))


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
    '''
    #plt.subplot(212)
    plt.figure()
    plt.title('$<\psi|\sigma_{z_{' + str(i + 1) + '}}|\psi>$')
    plt.plot(range(t_max), expectation_graph[i, k - 1, :], 'o')
    #plt.plot(range(t_max), np.ones(t_max) * expectation_wf[i][0], linewidth=2)
    plt.plot(range(t_max), expectation_mps[i, k - 1, :], linewidth=1)
    plt.ylim([-1.1, 1.1])
    plt.xticks(range(t_max))
    plt.ylabel('$m_z$')
    plt.xlabel('# of BP iterations')
    #plt.legend(['$<\sigma_z>_{DE-NFG}$', '$<\sigma_z>_{exact}$', '$<\sigma_z>_{MPS}$'])
    plt.legend(['$<\sigma_z>_{DE-NFG}$', '$<\sigma_z>_{MPS}$'])
    plt.grid()
    plt.show()

'''
for i in range(n):
    nrows, ncols = np.real(expectation_graph[0, :, :]).shape
    kk = np.linspace(0, k, ncols)
    tt = np.linspace(0, t_max, nrows)
    K, T = np.meshgrid(kk, tt)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.set_title(spins[i] + ' spin')
    ax.plot_surface(K, T, np.real(expectation_graph[i, :, :]), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.view_init(20, 30)
    ax.set_xlabel('bond dimension')
    ax.set_ylabel('# bp iterations')
    ax.set_zlabel('$<\sigma_z>$')
    ax.set_zlim([-1.1, 1.1])
    plt.show()
'''

