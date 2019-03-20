from scipy.stats import unitary_group
import matplotlib.pyplot as plt
import numpy as np
import DEnFG as denfg


t_max = 10
error = 1e-6
v_bethe = []
v_z = []
for j in range(10):
    g = denfg.Graph()
    alpha0 = 2
    alpha1 = 2
    alpha2 = 2
    alpha3 = 2

    g.add_node(alpha0, 'n0')
    g.add_node(alpha1, 'n1')
    g.add_node(alpha2, 'n2')
    g.add_node(alpha3, 'n3')

    #g.add_factor({'n0': 0, 'n1': 1}, np.complex128(np.ones((alpha0, alpha1)) + np.random.rand(alpha0, alpha1) + np.random.rand(alpha0, alpha1) * 1j))
    #g.add_factor({'n1': 0, 'n2': 1}, np.complex128(np.ones((alpha1, alpha2)) + np.random.rand(alpha1, alpha2) + np.random.rand(alpha1, alpha2) * 1j))
    #g.add_factor({'n2': 0, 'n3': 1}, np.complex128(np.ones((alpha2, alpha3)) + np.random.rand(alpha2, alpha3) + np.random.rand(alpha2, alpha3) * 1j))
    #g.add_factor({'n3': 0, 'n0': 1}, np.complex128(np.ones((alpha3, alpha0)) + np.random.rand(alpha3, alpha0) + np.random.rand(alpha3, alpha0) * 1j))
    #g.add_factor({'n0': 0, 'n2': 1}, np.complex128(np.ones((alpha0, alpha2)) + np.random.rand(alpha0, alpha2) + np.random.rand(alpha0, alpha2) * 1j))
    #g.add_factor({'n1': 0, 'n3': 1}, np.complex128(np.ones((alpha1, alpha3)) + np.random.rand(alpha1, alpha3) + np.random.rand(alpha1, alpha3) * 1j))

    g.add_factor({'n0': 0, 'n1': 1}, np.complex128(np.ones((alpha0, alpha1))))
    g.add_factor({'n1': 0, 'n2': 1}, np.complex128(np.ones((alpha1, alpha2))))
    g.add_factor({'n2': 0, 'n3': 1}, np.complex128(np.ones((alpha2, alpha3))))
    g.add_factor({'n3': 0, 'n0': 1}, np.complex128(np.ones((alpha3, alpha0))))
    g.add_factor({'n0': 0, 'n2': 1}, np.complex128(np.ones((alpha0, alpha2))))
    #g.add_factor({'n1': 0, 'n3': 1}, np.complex128(np.ones((alpha0, alpha2))))




    #g.add_factor({'n0': 0, 'n1': 1}, np.random.rand(alpha0, alpha1) + np.random.rand(alpha0, alpha1) * 1j)
    #g.add_factor({'n1': 0, 'n2': 1}, np.random.rand(alpha1, alpha2) + np.random.rand(alpha1, alpha2) * 1j)
    #g.add_factor({'n2': 0, 'n3': 1}, np.random.rand(alpha2, alpha3) + np.random.rand(alpha2, alpha3) * 1j)
    #g.add_factor({'n3': 0, 'n0': 1}, np.random.rand(alpha3, alpha0) + np.random.rand(alpha3, alpha0) * 1j)
    #g.add_factor({'n0': 0, 'n2': 1}, np.random.rand(alpha0, alpha2) + np.random.rand(alpha0, alpha2) * 1j)
    #g.add_factor({'n1': 0, 'n3': 1}, np.random.rand(alpha1, alpha3) + np.random.rand(alpha1, alpha3) * 1j)



    g.sum_product(t_max, error)
    g.calc_node_partition()
    g.calc_factor_partition()
    bethe = g.bethe_partition()
    z = g.calc_partition()
    v_z.append(z)
    v_bethe.append(bethe)
    g.calc_node_belief()
    g.calc_factor_belief()
    g.bethe_partition2()
    '''
    for n in g.nodes:
        for f in g.nodes[n][1]:
            data = np.empty((g.nodes[n][0], t_max), dtype=np.complex128)
            for t in range(t_max):
                for j in range(g.nodes[n][0]):
                    data[j, t] = g.all_messages[n][f][t][j, j]
            plt.figure()
            plt.title(n + ' to ' + f)
            for i in range(g.nodes[n][0]):
                plt.plot(range(t_max), np.real(data[i, :]), 'o')
            plt.ylim([0, 1])
            plt.yticks(np.linspace(0, 1, 11))
            plt.show()
            plt.pause(0.3)
    
    
    for f in g.factors:
        for n in g.factors[f][0]:
            data = np.empty((g.nodes[n][0], t_max), dtype=np.complex128)
            for t in range(t_max):
                for j in range(g.nodes[n][0]):
                    data[j, t] = g.all_messages[f][n][t][j, j]
            plt.figure()
            plt.title(f + ' to ' + n)
            for i in range(g.nodes[n][0]):
                plt.plot(range(t_max), np.real(data[i, :]), 'o')
            plt.ylim([0, 1])
            plt.yticks(np.linspace(0, 1, 11))
            plt.show()
            plt.pause(0.3)
    '''
plt.figure()
plt.plot(np.array(v_z), np.array(v_bethe), '.')
plt.plot([0, np.max(np.array(v_z))], [0, np.max(np.array(v_z))], '-.')
plt.axis('tight')
#lim = np.max(v_z)
#plt.ylim([0, lim])
#plt.xlim([0, lim])
#plt.xticks([0, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
#plt.yticks([0, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
plt.ylabel('$Z_{Bethe}$')
plt.xlabel('$Z$')
plt.show()


