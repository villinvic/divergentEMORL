import numpy as np


def log_uniform(low=0, high=1, size=None, base=10):
    assert low < high
    low = np.log(low + 1e-8)/np.log(base)
    high = np.log(high + 1e-8)/np.log(base)
    return np.power(base, np.random.uniform(low, high, size))

def uniform_with_hole(low=55, high=200):
    angle = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(low, high)
    x = np.cos(angle) * r
    y = np.sin(angle) * r
    return np.array([x,y], dtype=np.float32)

def kl_divergence(a, b):
    np.sum(a * np.log((a+1e-8)) - np.log((b+1e-8)))