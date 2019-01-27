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
y = 1j/np.sqrt(2) * np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]])


correlx = []
correlz = []
correlxz = []
mag_z = []
mag_x = []
mag_y = []

mag_z.append(aklt.OneSpinMeasurement(z, 0))
mag_x.append(aklt.OneSpinMeasurement(x, 0))
mag_y.append(aklt.OneSpinMeasurement(y, 0))

for i in range(1, n):
    print(i)
    mag_z.append(aklt.OneSpinMeasurement(z, i))
    mag_x.append(aklt.OneSpinMeasurement(x, i))
    mag_y.append(aklt.OneSpinMeasurement(y, i))
    correlx.append(aklt.TwoSpinsMeasurement(0, i, x, x))
    correlz.append(aklt.TwoSpinsMeasurement(0, i, z, z))
    correlxz.append(aklt.TwoSpinsMeasurement(0, i, x, z))


plt.figure()
plt.title('MAGNETIZATION')
plt.plot(range(n), mag_z, 'o')
plt.plot(range(n), mag_x, 'o')
plt.plot(range(n), mag_y, 'o')
plt.legend(['mag_z', 'mag_x', 'mag_y'])
plt.grid()
plt.show()

plt.figure()
plt.title('CORRELATIONS')
plt.plot(range(1, n), correlx)
plt.plot(range(1, n), correlz)
plt.plot(range(1, n), correlxz)
plt.legend(['<XX>', '<ZZ>', '<XZ>'])
plt.grid()
plt.show()











