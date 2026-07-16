'''
MSELoss, CrossEntropyLoss
Mean squared error for regression, Cross Entropy for Classification
'''

import numpy as np

class MSELoss:
    def forward(self, pred, y):
        self.pred = pred
        self.y = y
        return np.mean((pred - y) ** 2)

    def backward(self):
        n = self.pred.shape[0]
        return (2 / n) * (self.pred - self.y)

