from EMORL.misc import policy_similarity, normalize
from EMORL.Individual import Individual
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

np.set_printoptions(precision=4, suppress=True)

a = Individual(0, 60, 10, [], batch_dim=(20,1), trainable=True)
b = Individual(0, 60, 10, [], batch_dim=(20,1), trainable=True)
c = Individual(0, 60, 10, [], batch_dim=(20,1), trainable=True)
pop = [a, b, c]

a_w = a.genotype['brain'].get_training_params()
a_w['actor_core'][0][0][::2] += np.random.random(a_w['actor_core'][0][0][::2].shape)*0.1
a_w['actor_core'][0][1][::3] -= np.random.random(a_w['actor_core'][0][1][::3].shape)*0.1

b_w = b.genotype['brain'].get_training_params()
b_w['actor_core'][0][0][::5] -= np.random.random(b_w['actor_core'][0][0][::5].shape)*0
b_w['actor_core'][0][1][::1] -= np.random.random(b_w['actor_core'][0][1][::1].shape)*0

c.genotype['brain'].set_training_params(a_w)
b.genotype['brain'].set_training_params(b_w)

state = np.random.random((20,1,60)) * 0.1
state += np.random.normal(0,0.1, (20,1, 60))
out = np.array([i.probabilities_for(state) for i in pop])
ent = [-np.mean(np.sum(p * np.log(p), axis=-1)) for p in out]
flattened = out.reshape(3, 20*10)
normalized = normalize(flattened)
#print(normalized)
#print(out, ent, policy_similarity(*normalized[1:], l=5), policy_similarity(*normalized[:-1], l=5), policy_similarity(normalized[0], normalized[2], l=5))

K = rbf_kernel(normalized[:])
print(ent)
print(K, np.linalg.det(K))
