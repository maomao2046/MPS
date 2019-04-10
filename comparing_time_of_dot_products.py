import  numpy as np
import time
import matplotlib.pyplot as plt

f_dot = []
f_einsum = []
f_tensordot = []

#s = None
#e = None

for i in range(100000):
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    s = time.time()
    c = a.dot(b)
    e = time.time()
    f_dot.append(e - s)

    s = time.time()
    c = np.tensordot(a, b, 1)
    e = time.time()
    f_tensordot.append(e - s)

    s = time.time()
    c = np.einsum(a, [0, 1], b, [1, 2], [0, 2])
    e = time.time()
    f_einsum.append(e - s)

m_dot = np.mean(f_dot)
m_einsum = np.mean(f_einsum)
m_tensordot = np.mean(f_tensordot)

v_dot = np.var(f_dot)
v_einsum = np.var(f_einsum)
v_tensordot = np.var(f_tensordot)

plt.figure()
plt.plot(f_dot, '.')
plt.plot(f_einsum, '.')
plt.plot(f_tensordot, '.')
plt.legend(['f_dot', 'f_einsum', 'f_tensordot'])
plt.ylim([0, 0.004])
plt.show()

