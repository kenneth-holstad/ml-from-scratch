# Sequential

from .layers import Layer, Dense
import numpy as np

class Sequential(Layer):
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    # parameters() removed - add back later if optimizer split
