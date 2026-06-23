# -*- coding: utf-8 -*-
"""

"""

from utils.distances import euclidean_distances
from utils.preprocessing import StandardScaler
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train, self.y_train = X, y   # "training" = just storing data

    def predict(self, X):
        # compute distances, find k nearest, vote/average
        predictions = []
        for x in X:
            distances = euclidean_distances(x, self.X_train)
            k_indices = np.argost(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            predictions.append(np.bincount(k_labels).argmax())
        return np.array(predictions)