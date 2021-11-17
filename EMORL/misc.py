import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def log_uniform(low=0, high=1, size=None, base=10):
    assert low < high
    low = np.log(low + 1e-8)/np.log(base)
    high = np.log(high + 1e-8)/np.log(base)
    return np.power(base, np.random.uniform(low, high, size))

def uniform_with_hole(low=70, high=200):
    angle = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(low, high)
    x = np.cos(angle) * r
    y = np.sin(angle) * r
    return np.array([x,y], dtype=np.float32)

def kl_divergence(a, b):
    return np.mean(np.sum(a * (np.log(a+1e-8) - np.log(b+1e-8)), axis=-1))

def bc_coef(a, b):
    return np.sum(np.sqrt(a * b), axis=-1)

def bc_distance(a, b):
    return np.mean(-np.log(bc_coef(a, b)+1e-8))

def norm(a, b):
    return np.linalg.norm(a-b)

def normalize(x, clip=2):
    return np.clip((x - np.mean(x, axis=0))/(np.std(x, axis=0)+1e-8),-clip, clip)

def policy_similarity(a, b, l=1, func=norm):
    return np.exp(-func(a, b)**2/(2 * l ** 2))

def nn_crossover(a, b, architecture={}):
    pairs = [pairwise_cross_corr(ax, bx) for ax,bx in zip(a,b)]
    W_permuted = [np.zeros((2,)+ax.shape, dtype=np.float32) for ax in a]
    for layer_index, previous_index in architecture.items():
        if previous_index is not None:
            W_permuted[layer_index][:, :-1, :] = a[layer_index][pairs[previous_index][0], :], b[layer_index][pairs[previous_index][1], :]

        layer_crossover(a, b, pairs[layer_index], layer_index, previous_index, W_permuted[layer_index])

    offspring = [(safe_crossover(*np.rollaxis(W_permuted[d], 0))) for d in range(len(a))]

    return offspring


def layer_crossover(a, b, pairs, layer_index, previous_index, permuted):
    ax = a[layer_index]
    bx = b[layer_index]

    if previous_index is None:
        permuted[:] = ax[:, pairs[0]], bx[:, pairs[1]]
    else:
        permuted[:] = permuted[0][:, pairs[0]], permuted[1][:, pairs[1]]


def corr_matrix(X,Y):
    xvar = tf.reduce_sum(tf.math.squared_difference(X, tf.reduce_mean(X, axis=0)), axis=0)
    yvar = tf.reduce_sum(tf.math.squared_difference(Y, tf.reduce_mean(Y, axis=0)), axis=0)
    dot = tf.tensordot(tf.transpose(X), Y, axes=1)

    M = dot / tf.sqrt(xvar * yvar)

    return M


def pairwise_cross_corr(La, Lb):
    n = La.shape[1]
    scaler = StandardScaler()
    n_La = scaler.fit_transform(La)
    n_Lb = scaler.fit_transform(Lb)

    m = corr_matrix(n_La, n_Lb).numpy()

    m[np.isnan(m)] = -1
    argmax_columns = np.flip(np.argsort(m, axis=0), axis=0)
    #print(np.max(argmax_columns))
    #print(argmax_columns.shape)
    dead_neurons = np.sum(m, axis=0) == - n

    pairs = np.full((2, n), fill_value=np.nan, dtype=np.int32)
    pairs[1:] = np.arange(n)
    index_add = 0

    for index in range(n):
        if not dead_neurons[index]:
            for count in range(n):
                if argmax_columns[count, index] not in pairs[0]:
                    pairs[0, index_add] = argmax_columns[count, index]
                    index_add += 1
                    break

    for index in range(n):
        if index not in pairs[0] and index_add < n:
            pairs[0, index_add] = index
            index_add += 1

    #print(pairs)
    return pairs


def get_indice_pairs(ax, bx):
    l = np.empty((2,len(ax)), dtype=np.int32)
    for index in range(len(ax)):
        sm = np.abs(np.min(ax[index]) + np.min(bx[index]))
        sp = np.abs(np.max(ax[index]) + np.max(bx[index]))
        if sp > sm:
            l[0, index] = np.argmax(ax[index])
            l[1, index] = np.argmax(bx[index])
        else:
            l[0, index] = np.argmin(ax[index])
            l[1, index] = np.argmax(bx[index])

    return l


def safe_crossover(ax, bx):
    t = np.random.uniform(-0.25, 1.25)
    return (1-t) * ax + t * bx


class MovingAverage:
    def __init__(self, length, last_k=None):
        if last_k is None:
            self.last_k = int(length * 0.33)
        else:
            self.last_k = last_k

        self.length = length
        self.trend_count = 0
        self.index = 0
        self.values = np.full((length,), np.nan, dtype=np.float32)
        self.moving_avg = np.full((length,), np.nan, dtype=np.float32)

    def __call__(self):
        return self.moving_avg[(self.index-1)%self.length]

    def push(self, x):
        self.values[self.index % self.length] = x
        self.moving_avg[self.index % self.length] = np.nanmean(self.values)
        self.index += 1
        if self.index > self.last_k:
            trend = np.sign(self.trend())
            if trend != np.sign(self.trend_count):
                self.trend_count = 0
            self.trend_count += trend

    def reset(self):
        self.__init__(self.length)

    def trend(self):
        indexes = np.arange(self.last_k)
        values = self.values.take(np.arange((self.index%self.length)-self.last_k, self.index%self.length), mode='wrap')
        args = np.argwhere(np.logical_not(np.isnan(values))).flatten()
        return np.float32(np.polyfit(indexes[args], values[args], 1)[-2])

if __name__ == '__main__':

    from EMORL.Individual import Individual
    import os
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.summary.experimental.set_step(0)

    s = np.random.random((256, 80, 100))
    a = Individual(0, 100, 32, [], trainable=True)
    a_w = a.genotype['brain'].get_training_params()
    a_w['actor_core'][0][0][::15] += np.random.random(a_w['actor_core'][0][0][::15].shape)*0.9
    a_w['actor_core'][0][1][::3] -= np.random.random(a_w['actor_core'][0][1][::3].shape) *0.6
    a.genotype['brain'].set_training_params(a_w)
    b = Individual(0, 100, 32, [], trainable=True)
    c = Individual(0, 100, 32, [], trainable=True)

    c.inerit_from(a, b)
    a_out = a.genotype['brain'].policy.get_probs(a.genotype['brain'].dense_1(a.genotype['brain'].lstm(s)))
    b_out = b.genotype['brain'].policy.get_probs(b.genotype['brain'].dense_1(b.genotype['brain'].lstm(s)))
    c_out = c.genotype['brain'].policy.get_probs(c.genotype['brain'].dense_1(c.genotype['brain'].lstm(s)))

    for i, one in enumerate([a_out, b_out, c_out]):
        for j, two in enumerate([a_out, b_out, c_out]):
            if i != j and i < j:
                print((i, j), policy_similarity(one, two, 3, func=kl_divergence), policy_similarity(one, two, l=3, func=bc_distance))



    """
    
    for nn in [a,b,c]:
        nn.init_body(np.zeros((256,80,100)))

    a_w = a.get_training_params()
    a_w['actor_head'][1][:] += 1.0
    a_w['dense_1'][0][30:150] -= 0.1
    a.set_training_params(a_w)
    b_w = b.get_training_params()

    l = []
    for w in [a_w, b_w]:
        l.append([])
        for layer_name, weights in w.items():
            if 'core' in layer_name:
                for sub_layer in weights:
                    l[-1].append(sub_layer[0])
                    l[-1][-1] =  np.concatenate([l[-1][-1], sub_layer[1][np.newaxis]], axis=0)
            else:
                if 'lstm' in layer_name:
                    l[-1].append(weights[0][:, :weights[0].shape[1]*3//4])
                    l[-1].append(weights[0][:, weights[0].shape[1]*3//4:])
                    l[-1].append(weights[1])
                    l[-1][-3] = np.concatenate([l[-1][-3], weights[-1][np.newaxis,:weights[0].shape[1]*3//4]], axis=0)
                    l[-1][-2] = np.concatenate([l[-1][-2], weights[-1][np.newaxis, weights[0].shape[1] * 3 // 4:]],
                                               axis=0)
                else:
                    l[-1].append(weights[0])
                    l[-1][-1] =  np.concatenate([l[-1][-1], weights[1][np.newaxis]], axis=0)

    crossovered = nn_crossover(*l, architecture={
        0:None, 1:None, 2:None, 3:1, 4:3, 5:4, 6:3, 7:6
    })

    c_w = deepcopy(a_w)

    count = 0
    for layer_name, weights in c_w.items():
        print(layer_name)
        if 'core' in layer_name:
            for sub_layer_i in range(len(weights)):
                weights[sub_layer_i][0][:] = crossovered[count][:-1]
                weights[sub_layer_i][1][:] = crossovered[count][-1]
                count += 1
        else:
            if 'lstm' in layer_name:
                print(weights[0].shape)
                weights[0][:, :weights[0].shape[1] * 3 // 4] = crossovered[count][:-1]
                weights[-1][:weights[0].shape[1] * 3 // 4] = crossovered[count][-1]
                count += 1
                weights[0][:, weights[0].shape[1] * 3 // 4:] = crossovered[count][:-1]
                weights[-1][weights[0].shape[1] * 3 // 4:] = crossovered[count][-1]
                count += 1
                weights[1][:] = crossovered[count]
                count += 1

            else:
                weights[0][:] = crossovered[count][:-1]
                weights[1][:] = crossovered[count][-1]
                count += 1

    c.set_training_params(c_w)

    a_out = a.policy.get_probs(a.dense_1(a.lstm(s)))
    b_out = b.policy.get_probs(b.dense_1(b.lstm(s)))
    c_out = c.policy.get_probs(c.dense_1(c.lstm(s)))

    for i, one in enumerate([a_out, b_out, c_out]):
        for j, two in enumerate([a_out, b_out, c_out]):
            if i != j and i < j:
                print((i,j), policy_similarity(one,two, 1000))



    d = 8
    coefs = np.random.normal(size=d)
    for i in range(d):
        coefs[i]*= 0.005 ** i

    x = np.array([ sum([c * x**i for i, c in enumerate(coefs)]) for x in range(-150, 150)])
    x *= np.random.random(x.shape) + 0.1
    mving = MovingAverage(300)
    for i, d in enumerate(x):
        mving.push(d)
        print(i, mving.trend_count)

    print(mving(), mving.trend())

    import matplotlib.pyplot as plt

    plt.plot(x)
    plt.draw()
    plt.show()
    """


