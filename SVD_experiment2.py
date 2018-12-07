import numpy as np

d = 5
k = 3

psi = np.floor(np.random.randn(d, d, d, d))
u1, s1, v1 = np.linalg.svd(np.reshape(psi, (d, d ** 3)), full_matrices=True)
s1m = np.zeros([u1.shape[1], v1.shape[0]])
np.fill_diagonal(s1m, s1)
v1 = np.reshape(v1, (d ** 3, d, d, d))

u2, s2, v2 = np.linalg.svd(np.reshape(v1, (d ** 4, d ** 2)), full_matrices=True)
s2m = np.zeros([u2.shape[1], v2.shape[0]])
np.fill_diagonal(s2m, s2)
u2 = np.reshape(u2, (d ** 3, d, d ** 4))
v2 = np.reshape(v2, (d ** 2, d, d))

u3, s3, v3 = np.linalg.svd(np.reshape(v2, (d ** 3, d)), full_matrices=True)
s3m = np.zeros([u3.shape[1], v3.shape[0]])
np.fill_diagonal(s3m, s3)
u3 = np.reshape(u3, (d ** 2, d, d ** 3))
u4 = np.reshape(v3, (d, d))

psi_t = np.tensordot(u1, s1m, (len(np.shape(u1)) - 1, 0))
psi_t = np.tensordot(psi_t, u2, (len(np.shape(psi_t)) - 1, 0))
psi_t = np.tensordot(psi_t, s2m, (len(np.shape(psi_t)) - 1, 0))
psi_t = np.tensordot(psi_t, u3, (len(np.shape(psi_t)) - 1, 0))
psi_t = np.tensordot(psi_t, s3m, (len(np.shape(psi_t)) - 1, 0))
psi_t = np.tensordot(psi_t, u4, (len(np.shape(psi_t)) - 1, 0))


print('u1 shape - ', u1.shape)
print('s1m shape - ', s1m.shape)
print('u2 shape - ', u2.shape)
print('s2m shape - ', s2m.shape)
print('u3 shape - ', u3.shape)
print('s3m shape - ', s3m.shape)
print('u4 shape - ', u4.shape)
print('psi shape - ', psi.shape)
print('psi_t shape - ', psi_t.shape)


norm_of_psi = np.linalg.norm(psi)
norm_of_psi_t = np.linalg.norm(psi_t)
error = np.linalg.norm(psi - psi_t)

print('|psi| = ', norm_of_psi)
print('|psi_t| = ', norm_of_psi_t)
print('error = sqrt((psi - psi_t)^2)', error)




'''
# Working for three indices

psi = np.floor(np.random.randn(d, d, d))
u1, s1, v1 = np.linalg.svd(np.reshape(psi, (d, d * d)), full_matrices=True)
s1m = np.zeros([u1.shape[1], v1.shape[0]])
np.fill_diagonal(s1m, s1)
v1 = np.reshape(v1, (d ** 2, d, d))

psi_t = np.tensordot(u1, s1m, (len(np.shape(u1)) - 1, 0))
psi_t = np.tensordot(psi_t, v1, (len(np.shape(psi_t)) - 1, 0))

print('u1 shape - ', u1.shape)
print('s1m shape - ', s1m.shape)
print('v1 shape - ', v1.shape)
print('psi shape - ', psi.shape)
print('psi_t shape - ', psi_t.shape)


norm_of_psi = np.linalg.norm(psi)
norm_of_psi_t = np.linalg.norm(psi_t)
error = np.linalg.norm(psi - psi_t)

print('|psi| = ', norm_of_psi)
print('|psi_t| = ', norm_of_psi_t)
print('error = sqrt((psi - psi_t)^2)', error)
'''

