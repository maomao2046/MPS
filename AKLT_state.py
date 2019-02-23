import numpy as np
import matrix_product_state as mps
import matplotlib.pyplot as plt
import MPS as ms

# 1D AKLT
n = 6
spin1_tensor = np.array([[[1, 0], [0, 0]], [[0, 1/np.sqrt(2)], [1/np.sqrt(2), 0]], [[0, 0], [0, 1]]])
#spin1_tensor = np.array([[[0, np.sqrt(2. / 3)], [0, 0]], [[- np.sqrt(1. / 3), 0], [0, np.sqrt(1. / 3)]], [[0, 0], [- np.sqrt(2. / 3), 0]]])
singlet = np.array([[0, 1/np.sqrt(2)], [-1/np.sqrt(2), 0]])
aklt = mps.MPS('PBC')
for i in range(n):
    aklt.add_physical_tensor(spin1_tensor, 0)
    aklt.add_virtual_tensor(singlet)
z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
x = 1/np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
y = 1j/np.sqrt(2) * np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]])

norm = aklt.NormalizationFactor()

correlx = []
correlz = []
mag_z = []
mag_x = []
mag_y = []



for i in range(n):
    print(i)
    mag_z.append(aklt.OneSpinMeasurement(z, i) / norm)
    mag_x.append(aklt.OneSpinMeasurement(x, i) / norm)
    mag_y.append(aklt.OneSpinMeasurement(y, i) / norm)
    correlx.append(aklt.TwoSpinsMeasurement(0, i, x, x) / norm - (mag_x[i] * mag_x[i]))
    correlz.append(aklt.TwoSpinsMeasurement(0, i, z, z) / norm - (mag_z[i] * mag_z[i]))


plt.figure()
plt.title('MAGNETIZATION')
plt.plot(range(n), mag_z, 'o')
plt.plot(range(n), mag_x, 'o')
plt.plot(range(n), mag_y, '.')
plt.legend(['mag_z', 'mag_x', 'mag_y'])
plt.grid()
plt.show()

plt.figure()
plt.title('CORRELATIONS')
plt.plot(range(n), correlx, 'o')
plt.plot(range(n), correlz)
plt.legend(['<XX>', '<ZZ>'])

plt.grid()
plt.show()

# senity check
lspin = 3
rspin = 5
hij_0 = aklt.TwoSpinsMeasurement(lspin, rspin, np.eye(3), np.eye(3)) / norm
hij_x1 = aklt.TwoSpinsMeasurement(lspin, rspin, x, x) / norm
hij_y1 = aklt.TwoSpinsMeasurement(lspin, rspin, y, y) / norm
hij_z1 = aklt.TwoSpinsMeasurement(lspin, rspin, z, z) / norm
hij_xx = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(x, x), np.matmul(x, x)) / norm
hij_yy = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(y, y), np.matmul(y, y)) / norm
hij_zz = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(z, z), np.matmul(z, z)) / norm
hij_xy = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(x, y), np.matmul(x, y)) / norm
hij_yx = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(y, x), np.matmul(y, x)) / norm
hij_xz = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(x, z), np.matmul(x, z)) / norm
hij_zx = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(z, x), np.matmul(z, x)) / norm
hij_yz = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(y, z), np.matmul(y, z)) / norm
hij_zy = aklt.TwoSpinsMeasurement(lspin, rspin, np.matmul(z, y), np.matmul(z, y)) / norm

hij = 1. / 3 * hij_0 + 1. / 2 * (hij_x1 + hij_y1 +hij_z1) + 1. / 6 * (hij_xx + hij_yy + hij_zz + hij_xy + hij_yx + hij_yz + hij_zy + hij_xz + hij_zx)
print(hij)

print('\n')
print('############################################################################################################')
print('senity check is good')
print('$$$ need to rewrite the contraction over the PBC $$$')
print('\n')

psi = ms.mps2tensor(aklt.mps)