import functools
import numpy as np


def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = args + tuple(kwargs.items())
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

def sq_norm(x, axis=None):
    if axis is None:
        x = x.reshape(-1)
        return np.dot(x, x)
    else:
        return np.sum(x ** 2, axis=axis)