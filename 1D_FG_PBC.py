import numpy as np
import Loopy_Belief_Propagation as lbp
import matplotlib.pyplot as plt

n = range(2, 50)
t_max = 50
alphabet = 2
epsilon = 1e-6
tensor = np.arange(4.).reshape(2, 2)
first_node_belief = np.zeros((alphabet, len(n)))

for i in range(len(n)):
    print(i)
    # name of graph
    N = n[i]
    g = lbp.Graph(N)

    # nodes
    for j in range(N):
        g.add_node(alphabet, 'n' + str(j))

    # interactions
    for j in range(N - 1):
        g.add_factor({'n' + str(j): 0, 'n' + str(j + 1): 1}, tensor)
    g.add_factor({'n' + str(N - 1): 0, 'n' + str(0): 1}, tensor)  # periodic BC
    g.sum_product(t_max, epsilon)
    g.beliefs()
    first_node_belief[:, i] = g.node_belief['n1']

plt.figure()
plt.plot(n, first_node_belief[0, :], 'o')
plt.plot(n, first_node_belief[1, :], 'o')
plt.grid()
plt.show()



