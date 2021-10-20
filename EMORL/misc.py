import numpy as np


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
    return np.sum(a * (np.log(a+1e-8) - np.log(b+1e-8)))

def policy_similarity(a, b, l=1):
    return np.exp(-kl_divergence(a, b)**2/(2 * l ** 2))


def nn_crossover(a, b):
    offspring = []
    Wd = None
    for d, (ax, bx) in enumerate(zip(a,b)):
        Wd, Wdm1 = layer_crossover(ax, bx, d, Wd)
        offspring.append(safe_crossover(*np.rollaxis(Wdm1, 0)))


    return offspring


def layer_crossover(ax, bx, d, Wd=None):
    if Wd is None:
        Wd = np.empty((2,)+ax.shape, dtype=np.float32)
    Wdp1 = np.empty((2,)+ax.shape, dtype=np.float32)
    l = get_indice_pairs(ax, bx)
    if d == 0:
        print(Wd.shape, ax[:, l[0]].shape)
        Wd[:] = ax[:, l[0]], bx[:, l[1]]
    else:
        print(Wd.shape, l.shape)
        Wd[:] = Wd[:, l]
    Wdp1[:] = ax[l, :]

    return Wdp1, Wd



def get_indice_pairs(ax, bx):
    l = np.empty((2, len(ax)), dtype=np.int32)
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


if __name__ == '__main__':
    a = []
    b = []
    for _ in range(3):
        a.append(np.random.uniform(-1,1, size=(257, 256)))
        b.append(np.random.uniform(-1, 1, size=(257, 256)))


    c = nn_crossover(a,b)

    print(a,b,c)