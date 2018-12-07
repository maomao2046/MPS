import numpy as np

'''
MPS implementation Tensor Network
'''
d = 3
k = 2

psi = np.floor(np.random.randn(d, d, d, d, d, d))
u = {}
s = {}
v = {}

l = len(psi.shape)

for i in range(l - 1):
    if i == 0:
        new_shape = (d, d ** (l - 1))
        u0, s0, v0 = np.linalg.svd(np.reshape(psi, new_shape), full_matrices=True)
        # Truncation

        s[i] = np.zeros([u0.shape[1], v0.shape[0]])
        np.fill_diagonal(s[i], s0)
        u[i] = u0
        v_shape = np.ones(l, dtype=int) * d
        v_shape[0] = v_shape[0] ** (l - 1)
        v[i] = np.reshape(v0, v_shape)

    else:
        new_shape = (d ** (l + 1 - i), d ** (l - 1 - i))
        u0, s0, v0 = np.linalg.svd(np.reshape(v[i - 1], new_shape), full_matrices=True)
        # Truncation

        s[i] = np.zeros([u0.shape[1], v0.shape[0]])
        np.fill_diagonal(s[i], s0)
        u[i] = np.reshape(u0, (d ** (l - i), d, d ** (l + 1 - i)))
        v[i] = v0
u[i + 1] = v0

psi_t = np.tensordot(u[0], s[0], (len(np.shape(u[0])) - 1, 0))
for i in range(1, l - 1):
    psi_t = np.tensordot(psi_t, u[i], (len(np.shape(psi_t)) - 1, 0))
    psi_t = np.tensordot(psi_t, s[i], (len(np.shape(psi_t)) - 1, 0))
    print('u[' + str(i) + '] shape - ', u[i].shape)
    print('s[' + str(i) + '] shape - ', s[i].shape)
i += 1
psi_t = np.tensordot(psi_t, u[i], (len(np.shape(psi_t)) - 1, 0))
print('u[' + str(i) + '] shape - ', u[i].shape)

norm_of_psi = np.linalg.norm(psi)
norm_of_psi_t = np.linalg.norm(psi_t)
error = np.linalg.norm(psi - psi_t)

print('|psi| = ', norm_of_psi)
print('|psi_t| = ', norm_of_psi_t)
print('error = sqrt((psi - psi_t)^2)', error)

