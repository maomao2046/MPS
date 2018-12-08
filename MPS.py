import numpy as np

'''
 matrix_product_state is a function which given the tensor coefficient
 of an N spins with dimension d wave function would convert this tensor
 into a matrix product state (MPS) tensor network (TN) with truncation 
 variable 'k' (the approximation index)
 
'''


def matrix_product_state(psi, k):

    n = len(psi.shape)
    d = psi.shape[0]
    psi_shape = psi.shape
    if np.sum(np.array(psi_shape) / d) != n:
        raise IndexError('psi has different index sizes')

    u = {}
    s = {}
    v = {}
    mps = {}

    for i in range(n - 1):
        if i == 0:
            new_shape = (d, d ** (n - 1))
            u0, s0, v0 = np.linalg.svd(np.reshape(psi, new_shape), full_matrices=True)
            # Truncation

            s[i] = np.zeros([u0.shape[1], v0.shape[0]])
            np.fill_diagonal(s[i], s0)
            u[i] = u0
            v_shape = np.ones(n, dtype=int) * d
            v_shape[0] = v_shape[0] ** (n - 1)
            v[i] = np.reshape(v0, v_shape)

        else:
            new_shape = (d ** (n + 1 - i), d ** (n - 1 - i))
            u0, s0, v0 = np.linalg.svd(np.reshape(v[i - 1], new_shape), full_matrices=True)
            # Truncation

            s[i] = np.zeros([u0.shape[1], v0.shape[0]])
            np.fill_diagonal(s[i], s0)
            u[i] = np.reshape(u0, (d ** (n - i), d, d ** (n + 1 - i)))
            v[i] = v0
    u[i + 1] = v0
    j = 0
    for i in range(0, 2 * n - 2, 2):
        mps[i] = u[j]
        mps[i + 1] = s[j]
        j += 1
    mps[2 * n - 2] = u[n - 1]

    return mps

