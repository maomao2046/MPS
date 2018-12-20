import numpy as np
import MPS as MPS
import matplotlib.pyplot as plt

d = 2
n = 10
epsilon = 1e-2
spins = tuple(np.ones(n, dtype=int) * d)

'''
# GHZ + noise
ghz = np.zeros(spins, dtype=float)
ghz[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1 / np.sqrt(2)
ghz[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] = 1 / np.sqrt(2)
noise = np.floor(np.random.randn(*spins))
psi = epsilon * noise + ghz
'''

'''
psi = np.zeros(spins, dtype=float)
psi[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1
psi[0, 1, 0, 0, 0, 0, 0, 0, 0, 0] = 1
psi[0, 0, 1, 0, 0, 0, 0, 0, 0, 0] = 1
psi[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] = 1
psi[0, 0, 0, 0, 1, 0, 0, 0, 0, 0] = 1
psi[0, 0, 0, 0, 0, 1, 0, 0, 0, 0] = 1
psi[0, 0, 0, 0, 0, 0, 1, 0, 0, 0] = 1
psi[0, 0, 0, 0, 0, 0, 0, 1, 0, 0] = 1
psi[0, 0, 0, 0, 0, 0, 0, 0, 1, 0] = 1
psi[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] = 1
#s = 'psi = |100...0> + |010...0> + ... + |000...1>'
'''

psi = np.floor(np.random.randn(*spins))
psi /= np.linalg.norm(psi)

norm_of_psi_t = []
error = []
k = []

for i in range(0, np.int(np.floor(d ** (n / 2 - 1)))):

    k.append(i)
    #mps = MPS.canon_matrix_product_state(psi, k[i])
    mps = MPS.matrix_product_state(psi, k[i])
    psi_t = np.tensordot(mps[0], mps[1], (len(np.shape(mps[0])) - 1, 0))
    print('u0 shape - ', mps[0].shape)
    print('s0 shape - ', mps[1].shape)

    ju = 1
    js = 1
    for i in range(2, 2 * n - 1):
        psi_t = np.tensordot(psi_t, mps[i], (len(np.shape(psi_t)) - 1, 0))
        if not np.mod(i, 2):
            print('u' + str(ju) + ' shape - ', mps[i].shape)
            ju += 1
        if np.mod(i, 2):
            print('s' + str(js) + ' shape - ', mps[i].shape)
            js += 1

    norm_of_psi = np.linalg.norm(psi)
    norm_of_psi_t.append(np.linalg.norm(psi_t))
    error.append(np.linalg.norm(psi - psi_t))


plt.figure()
plt.title('MPS vs True wave function')
plt.plot(k, norm_of_psi * np.ones(len(k)), 'y')
plt.plot(k, norm_of_psi_t, '.')
plt.plot(k, error, '.')
plt.legend(['norm of psi', 'norm of psi mps', 'error'])
plt.xlabel('max # of eigenvalues not zero')
plt.ylabel('Frobenius norm')
plt.show()



