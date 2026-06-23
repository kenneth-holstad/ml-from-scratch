# -*- coding: utf-8 -*-
"""

"""

import numpy as np

def train_test_split(X, y, test_size=0.2, seed=None):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]