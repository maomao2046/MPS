import numpy as np
import matrix_product_state as mps
import matplotlib.pyplot as plt
import wavefunction as wf


# 1D AKLT
n = 4
spin1_tensor = np.array([[[1, 0], [0, 0]], [[0, 1/np.sqrt(2)], [1/np.sqrt(2), 0]], [[0, 0], [0, 1]]])
#spin1_tensor = np.array([[[0, np.sqrt(2. / 3)], [0, 0]], [[- np.sqrt(1. / 3), 0], [0, np.sqrt(1. / 3)]], [[0, 0], [- np.sqrt(2. / 3), 0]]])
singlet = np.array([[0, 1/np.sqrt(2)], [-1/np.sqrt(2), 0]])
aklt = mps.MPS('PBC')
for i in range(n):
    aklt.add_physical_tensor(spin1_tensor, 0)
    aklt.add_virtual_tensor(singlet)
z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
x = 1./np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
y = 1j/np.sqrt(2) * np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]])


psi = aklt.mps2tensor()
wf = wf.wavefunction()
wf.addwf(psi, n, 3)

expectation_x0_wf = wf.SingleSpinMeasurement(0, x)
expectation_x1_wf = wf.SingleSpinMeasurement(1, x)
expectation_x2_wf = wf.SingleSpinMeasurement(2, x)
expectation_x3_wf = wf.SingleSpinMeasurement(3, x)
#expectation_x4_wf = wf.SingleSpinMeasurement(4, x)
#expectation_x5_wf = wf.SingleSpinMeasurement(5, x)


expectation_z0_wf = wf.SingleSpinMeasurement(0, z)
expectation_z1_wf = wf.SingleSpinMeasurement(1, z)
expectation_z2_wf = wf.SingleSpinMeasurement(2, z)
expectation_z3_wf = wf.SingleSpinMeasurement(3, z)
#expectation_z4_wf = wf.SingleSpinMeasurement(4, z)
#expectation_z5_wf = wf.SingleSpinMeasurement(5, z)



norm = aklt.NormalizationFactor()

correlx = []
correlz = []
correlxn = []
correlzn = []
mag_z = []
mag_x = []
mag_y = []

mag_zn = []
mag_xn = []
mag_yn = []
norm_n = aklt.SingleSpinMeasure(np.eye(3), 0)


for i in range(n):
    print(i)
    mag_z.append(aklt.OneSpinMeasurement(z, i) / norm)
    mag_x.append(aklt.OneSpinMeasurement(x, i) / norm)
    mag_y.append(aklt.OneSpinMeasurement(y, i) / norm)
    mag_xn.append(aklt.SingleSpinMeasure(x, i) / norm_n)
    mag_zn.append(aklt.SingleSpinMeasure(z, i) / norm_n)
    mag_yn.append(aklt.SingleSpinMeasure(y, i) / norm_n)
    correlx.append(aklt.TwoSpinsMeasurement(0, i, x, x) / norm - (mag_x[i] * mag_x[i]))
    correlz.append(aklt.TwoSpinsMeasurement(0, i, z, z) / norm - (mag_z[i] * mag_z[i]))
    correlxn.append(aklt.Correlations(0, i, x, x) / norm_n - (mag_xn[i] * mag_xn[i]))
    correlzn.append(aklt.Correlations(0, i, z, z) / norm_n - (mag_zn[i] * mag_zn[i]))

plt.figure()
plt.title('MAGNETIZATION')
plt.plot(range(n), mag_z, 'o')
plt.plot(range(n), mag_x, 'o')
plt.plot(range(n), np.real(mag_y), '.')
plt.plot(range(n), mag_zn, 's')
plt.plot(range(n), mag_xn, 'v')
plt.plot(range(n), np.real(mag_yn), 'v')
plt.legend(['mag_z', 'mag_x', 'mag_y', 'mag_zn', 'mag_xn', 'mag_yn'])
plt.grid()
plt.show()

plt.figure()
plt.title('CORRELATIONS')
plt.plot(range(n), correlx, 'o')
plt.plot(range(n), correlz, 'o')
plt.plot(range(n), correlxn, 'v')
plt.plot(range(n), correlzn, 'v')
plt.legend(['<XX>', '<ZZ>', '<XX>n', '<ZZ>n'])
plt.grid()
plt.show()

'''
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
'''

