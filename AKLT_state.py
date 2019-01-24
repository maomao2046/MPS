import numpy as np
import MPS as MPS

aklt_1D_mps = {}
spin1_tensor = np.array([[[1, 0], [0, 0]], [[0, 1/np.sqrt(2)], [1/np.sqrt(2), 0]], [[0, 0], [0, 1]]])
singlet = np.array([[0, 1/np.sqrt(2)], [-1/np.sqrt(2), 0]])
n = 10
aklt_1D_mps[0] = spin1_tensor
aklt_1D_mps[1] = singlet

for i in range(2, 2 * n):
    if np.mod(i - 1, 2):
        aklt_1D_mps[i] = spin1_tensor
    if np.mod(i, 2):
        aklt_1D_mps[i] = singlet

psi_t = np.array(aklt_1D_mps[0])
for i in range(1, 2 * n):
    if len(np.shape(aklt_1D_mps[i])) == 3:
        psi_t = np.tensordot(psi_t, aklt_1D_mps[i], (len(np.shape(psi_t)) - 1, 1))
    else:
        psi_t = np.tensordot(psi_t, aklt_1D_mps[i], (len(np.shape(psi_t)) - 1, 0))

psi_t = np.trace(psi_t, 1, len(psi_t.shape) - 1)



