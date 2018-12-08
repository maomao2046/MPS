import numpy as np
import MPS as MPS

d = 3
k = 2
n = 6
spins = tuple(np.ones(n, dtype=int) * d)

psi = np.floor(np.random.randn(d, d, d, d, d, d))
mps = MPS.matrix_product_state(psi, k)
psi_t = np.tensordot(mps[0], mps[1], (len(np.shape(mps[0])) - 1, 0))
print('u0 shape - ', mps[0].shape)
print('s0 shape - ', mps[1].shape)

ju = 1
js = 2
for i in range(2, 2 * n - 1):
    psi_t = np.tensordot(psi_t, mps[i], (len(np.shape(psi_t)) - 1, 0))
    if not np.mod(i, 2):
        print('u' + str(ju) + ' shape - ', mps[i].shape)
        ju += 1
    if np.mod(i, 2):
        print('s' + str(js) + ' shape - ', mps[i].shape)
        js += 1

norm_of_psi = np.linalg.norm(psi)
norm_of_psi_t = np.linalg.norm(psi_t)
error = np.linalg.norm(psi - psi_t)

print('|psi| = ', norm_of_psi)
print('|psi_t| = ', norm_of_psi_t)
print('error = sqrt((psi - psi_t)^2)', error)