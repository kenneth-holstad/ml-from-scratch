# Parameter, Layer, Dense, ReLU, Sigmoid, Convo2D...

from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    """Anything that can take input forward and pass gradients backward."""
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        ...

class Dense(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.x.T @ grad_output
        self.grad_b = grad_output.sum(axis=0)
        return grad_output @ self.W.T

    def zero_grad(self): # comes from optimizer - remove later if split
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

