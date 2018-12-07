import numpy as np
import matplotlib.pyplot as plt

d = 10
khi = 4
epsilon = 1e-6

psi = np.expand_dims(np.expand_dims(np.eye(d), axis=1), axis=2) + epsilon * np.floor(np.random.randn(d, d, d, d))
u1, s1, vh1 = np.linalg.svd(np.reshape(psi, (d, d * d * d)), full_matrices=True)
u1_t = u1[:, 0:min(khi, d)]
s1_t = np.diag(s1[0:min(khi, d)])
vh1_t = vh1[0:min(khi, d), :]

vh1_t = np.reshape(vh1_t, (min(khi, d), d, d, d))

u2, s2, vh2 = np.linalg.svd(np.reshape(vh1_t, (min(khi, d), d, d * d)), full_matrices=True)
u2_t = u2[:, :, 0:min(khi ** 2, d)]
s2 = np.flipud(np.sort(np.reshape(s2, np.prod(np.shape(s2)))))
s2_t = np.diag(np.reshape(s2[0:min(khi ** 2, d)], min(khi ** 2, d)))
vh2 = np.reshape(vh2, (np.shape(vh2)[0] * np.shape(vh2)[1], np.shape(vh2)[2]))
vh2_t = np.reshape(vh2[0:min(khi ** 2, d), :], (min(khi ** 2, d), d, d))


u3, s3, vh3 = np.linalg.svd(vh2_t, full_matrices=True)
u3_t = u3[:, :, 0:min(khi ** 3, d)]
s3 = np.flipud(np.sort(np.reshape(s3, np.prod(np.shape(s3)))))
s3_t = np.diag(np.reshape(s3[0:min(khi ** 3, d)], min(khi ** 3, d)))
vh3 = np.reshape(vh3, (np.shape(vh3)[0] * np.shape(vh3)[1], np.shape(vh3)[2]))
vh3_t = np.reshape(vh3[0:min(khi ** 3, d), :], (min(khi ** 3, d), d))

u4_t = vh3_t

print('u1_t - ', np.shape(u1_t))
print('s1_t - ', np.shape(s1_t))
print('u2_t - ', np.shape(u2_t))
print('s2_t - ', np.shape(s2_t))
print('u3_t - ', np.shape(u3_t))
print('s3_t - ', np.shape(s3_t))
print('u4_t - ', np.shape(u4_t))

psi_t = np.tensordot(u1_t, s1_t, (len(np.shape(u1_t)) - 1, 0))
psi_t = np.tensordot(psi_t, u2_t, (len(np.shape(psi_t)) - 1, 0))
psi_t = np.tensordot(psi_t, s2_t, (len(np.shape(psi_t)) - 1, 0))
psi_t = np.tensordot(psi_t, u3_t, (len(np.shape(psi_t)) - 1, 0))
psi_t = np.tensordot(psi_t, s3_t, (len(np.shape(psi_t)) - 1, 0))
psi_t = np.tensordot(psi_t, u4_t, (len(np.shape(psi_t)) - 1, 0))

print('psi - ', np.shape(psi))

norm_of_psi = np.linalg.norm(psi)
norm_of_psi_t = np.linalg.norm(psi_t)
error = np.linalg.norm(psi - psi_t)

print('|psi| = ', norm_of_psi)
print('|psi_t| = ', norm_of_psi_t)
print('error = ', error)
