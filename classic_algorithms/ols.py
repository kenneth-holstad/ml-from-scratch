"""
This is the closed-form solution to linear regression problems
Should generalize for both SLR and MLR (one or more predictors)

For gradient descent method use the neural network framework:

model = Sequential([Dense(n_features, 1)])
loss = MSELoss()
"""

import numpy as np

class OLS:
    def __init__(self):
        pass

    def fit(self, X, y):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])  # bias
        self.beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b @ self.beta
