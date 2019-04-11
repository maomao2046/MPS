import numpy as np
from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.pyplot as plt
import wavefunction as wf
import matrix_product_state as mps
import DEnFG as denfg
import time


k = 5
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
    b = wf.SingleSpinMeasurement(i, z)
    expectation_wf.append(b[0, 0] + 0.05)

expectation_mps1 = np.zeros(n, dtype=complex)
expectation_mps2 = np.zeros(n, dtype=complex)


model1 = mps.MPS('OPEN')
s = time.time()
model1.wavefunction2mps(wf.tensor, k)
e = time.time()
print(e - s, '  1')
s = time.time()
model2 = mps.MPS('OPEN')
e = time.time()
print(e - s, '  2')
model2.wavefunction2mps2(wf.tensor, k)
for i in range(n):
    expectation_mps1[i] = model1.SingleSpinMeasure(z, i)
    expectation_mps2[i] = model2.SingleSpinMeasure(z, i)


plt.figure()
plt.plot(range(n), expectation_mps1, 'o')
plt.plot(range(n), expectation_mps2, 'o')
plt.plot(range(n), expectation_wf, 'o')
plt.legend(['old', 'new', 'wf'])
plt.show()