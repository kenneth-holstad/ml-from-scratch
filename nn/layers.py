# Parameter, Layer, Dense, ReLU, Sigmoid, Convo2D...

class Layer(ABC):
    """Anything that can take input forward and pass gradients backward."""
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        ...

    def parameters(self) -> list[Parameter]:
        """Layers with no weights (activations) just return []."""
        return []

class Dense(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.W = Parameter(np.random.randn(in_features, out_features) * 0.01)
        self.b = Parameter(np.zeros(out_features))

    def forward(self, x):
        self.x = x
        return x @ self.W.data + self.b.data

    def backward(self, grad_output):
        self.W.grad += self.x.T @ grad_output
        self.b.grad += grad_output.sum(axis=0)
        return grad_output @ self.W.data.T

    def parameters(self):
        return [self.W, self.b]

