# -*- coding: utf-8 -*-
"""
Calculate distances needed for other algorithms
Starting with euclidean
"""

import numpy as np

# this assumes both X and Y are 2D
def euclidean_distances(X, Y):
    # this is so it doesn't break if you feed in a 1-D vector
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))

