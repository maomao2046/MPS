import numpy as np
import matrix_product_state as mps
import matplotlib.pyplot as plt

# 1D AKLT
n = 100
spin1_tensor = np.array([[[1, 0], [0, 0]], [[0, 1/np.sqrt(2)], [1/np.sqrt(2), 0]], [[0, 0], [0, 1]]])
singlet = np.array([[0, 1/np.sqrt(2)], [-1/np.sqrt(2), 0]])
aklt = mps.MPS('PBC')
for i in range(n):
    aklt.add_physical_tensor(spin1_tensor, 0)
    aklt.add_virtual_tensor(singlet)
z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
x = 1/np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

correlx = []
correlz = []
mag_z = []
mag_x = []
mag_z.append(aklt.OneSpinMeasurement(z, 0))
mag_x.append(aklt.OneSpinMeasurement(x, 0))
for i in range(1, n):
    mag_z.append(aklt.OneSpinMeasurement(z, i))
    mag_x.append(aklt.OneSpinMeasurement(x, i))
    correlx.append(aklt.TwoSpinsMeasurement(0, i, x, x))
    correlz.append(aklt.TwoSpinsMeasurement(0, i, z, z))

plt.figure()
plt.title('MAGNETIZATION')
plt.plot(range(n), mag_z, 'o')
plt.plot(range(n), mag_x, 'o')
plt.legend(['mag_z', 'mag_x'])
plt.show()

plt.figure()
plt.title('CORRELATIONS')
plt.plot(range(1, n), correlx)
plt.plot(range(1, n), correlz)
plt.legend(['XX', 'ZZ'])
plt.show()











