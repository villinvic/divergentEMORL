from EMORL.misc import policy_similarity, kl_divergence
import numpy as np

np.set_printoptions(precision=4, suppress=True)

full_dim = (1, 79, 8)
amount = 1
for d in full_dim[:-1]:
    amount *= d



policies = np.zeros( (10,)+full_dim, dtype=np.float64)
x = np.random.random(full_dim)
for i in range(10):
    policies[i, :] = x + i
    policies[i, :] /= np.sum(policies[i, :], axis=1)[:,np.newaxis]

K = np.zeros((10,10), dtype=np.float32)
kl = np.zeros((10,10), dtype=np.float32)

policies[:] = np.clip((policies - np.mean(policies, axis=0))/(np.std(policies, axis=0)+1e-8),-2, 2)




for i in range(10):
    for j in range(10):
        if i==j:
            K[i, j] = 1.
        elif j>i:
            K[i, j] = policy_similarity(policies[i], policies[j], l=8)
        else:
            K[i, j] = K[j, i] #policy_similarity(policies[i], policies[j], l=1)
        kl[i, j] = 1 - K[i, j]

#K = policy_similarity(policies[:,0,:], l=1)

print(K)
print(np.linalg.det(K), np.linalg.slogdet(K))
