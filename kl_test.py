from EMORL.misc import policy_similarity, kl_divergence
import numpy as np

np.set_printoptions(precision=4, suppress=True)

full_dim = (64, 79, 9)
amount = 1
for d in full_dim[:-1]:
    amount *= d



policies = np.zeros( (10,)+full_dim, dtype=np.float64)
x = np.random.random(full_dim)
for i in range(10):
    policies[i, :, :, :] = x + np.random.random(full_dim) * i * 2
    policies[i, :, :, :] /= np.sum(policies[i, :, :, :], axis=2)[:,:,np.newaxis]

K = np.zeros((10,10), dtype=np.float32)
kl = np.zeros((10,10), dtype=np.float32)

for i in range(10):
    for j in range(10):
        K[i, j] = policy_similarity(policies[i], policies[j], l=1500)
        kl[i, j] = 1 - K[i, j]

print(K)
print( np.linalg.det(K), np.linalg.slogdet(K))
