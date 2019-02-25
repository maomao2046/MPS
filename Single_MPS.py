import numpy as np
import MPS as MPS
import matplotlib.pyplot as plt

d = 2
n = 3
k = 200
#k = np.int(d ** (np.floor(n / 2)))
spins = tuple(np.ones(n, dtype=int) * d)

psi = np.array([[1., -1.j], [0., 1.]]) / np.sqrt(3)

#psi = np.random.randn(*spins) + 1j * np.random.randn(*spins)
#psi /= np.linalg.norm(psi)
rho = MPS.densitymat(psi)
plt.imshow(np.real(rho))
plt.show()
norm_of_psi_t = 0
error = 0


mps = MPS.canon_matrix_product_state(psi, k)

norm_of_psi = np.linalg.norm(psi)
#norm_of_psi_t = (np.linalg.norm(psi_t))
#error = (np.linalg.norm(psi - psi_t))

psi_new = MPS.mps2tensor(mps)
print(psi - psi_new)
print('norm_of_psi = ', norm_of_psi)
#print('norm_of_psi_t = ', norm_of_psi_t)
#print('error = ', error)

