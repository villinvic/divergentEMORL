from EMORL.misc import kl_divergence
import numpy as np

full_dim = (30, 15, 64, 79, 9)
amount = 1
for d in full_dim[:-1]:
    amount *= d

x = np.random.random(full_dim)
y = np.random.random(full_dim)**10
x /= np.sum(x, axis=4)[:,:,:,:,np.newaxis]
y /= np.sum(y, axis=4)[:,:,:,:,np.newaxis]

x[:,:,:,:,:] = 0.

print(x, y, np.max(x), np.max(y), np.sum(x), np.sum(y))


kl = kl_divergence(x,y)
print(kl/amount)