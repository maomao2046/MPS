import numpy as np
from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.pyplot as plt
import wavefunction as wf
import matrix_product_state as mps
import DEnFG as denfg
import time


k = 20
n = 10
alphabet = 2

'''
psi = np.array([[[[[[0., 0.], [0., 1.j]], [[0., 0.j], [0.j, 0.]]],
                [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [-1.j, 0.]]],
                 [[[0., 1.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                [[[[[0.j, 0.], [0.j, 0.j]], [[-1., 0.j], [-1.j, 0.]]],
                [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]],
                 [[[0., 1.j], [0., -1.j]], [[0.j, 0.], [0., 0.j]]]]]])

psi = np.array([[[[[[-0.5, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]],
                [[[[1.j, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.j]]]]],
                [[[[[0.2j, 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [-0.5, 0.j]], [[0.j, 0.], [0., 0.j]]]],
                 [[[[1., 0.], [0., 0.j]], [[0., 0.j], [0.j, 0.]]], [[[0., 0.j], [0., 0.j]], [[0.j, 0.], [0., 0.]]]]]])

'''
error1 = []
error2 = []
wf = wf.wavefunction()
wf.random_wavefunction(n, alphabet)
expectation_wf = []
z = np.array([[1, 0], [0, -1]])

for i in range(n - 1):
    b = wf.SingleSpinMeasurement(i, z)
    expectation_wf.append(b[0, 0])

for i in range(1, k):
    print(i)
    #wf.addwf(psi, n, alphabet)
    #wf.random_wavefunction(n, alphabet)

    expectation_mps1 = np.zeros(n - 1, dtype=complex)
    expectation_mps2 = np.zeros(n - 1, dtype=complex)


    model1 = mps.MPS('OPEN')
    s = time.time()
    model1.wavefunction2mps(wf.tensor, i)
    e = time.time()
    print(e - s, '  1')
    model2 = mps.MPS('OPEN')
    s = time.time()
    model2.wavefunction2mps2(wf.tensor, i)
    e = time.time()
    print(e - s, '  2')
    for j in range(n - 1):
        expectation_mps1[j] = model1.SingleSpinMeasure(z, j)
        expectation_mps2[j] = model2.SingleSpinMeasure(z, j)
    error1.append(np.sum(np.abs(np.array(expectation_wf) - np.array(expectation_mps1))))
    error2.append(np.sum(np.abs(np.array(expectation_wf) - np.array(expectation_mps2))))

    plt.figure()
    plt.plot(range(n - 1), expectation_mps1, 'o')
    plt.plot(range(n - 1), expectation_mps2, 'o')
    plt.plot(range(n - 1), expectation_wf, 'o')
    plt.legend(['old', 'new', 'wf'])
    plt.show()

z = np.polyfit(np.array(range(1, k)), np.log(np.array(error2)), 1)
plt.figure()
plt.plot(range(1, k), error1, 'o')
plt.plot(range(1, k), error2, 'o')
plt.plot(range(1, k), z[0] * np.exp(z[1] * np.array(range(1, k))))
plt.legend(['error1', 'error2'])
plt.show()