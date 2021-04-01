import numpy as np


class Param:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
        self.velocity = np.zeros_like(data)


class Linear:
    def __init__(self, input_size, output_size):
        self.W = Param(np.random.randn(input_size, output_size))
        self.B = Param(np.random.randn(output_size))
        self.params = [self.W, self.B]
        self.X = None

    def __call__(self, X):
        self.X = X
        return X @ self.W.data + self.B.data

    def backward(self, out):
        self.W.grad += (self.X.T @ out) / self.X.shape[0]
        self.B.grad += np.sum(out, axis=0) / self.X.shape[0]
        return out @ self.W.data.T

    def parameters(self):
        for p in self.params:
            yield p


class Sigmoid:
    def __init__(self):
        self.res = None

    def __call__(self, X):
        self.res = 1 / (1 + np.exp(-X))
        return self.res

    def backward(self, out):
        return (self.res * (1 - self.res)) * out

    def parameters(self):
        return None


class ReLU:
    def __init__(self):
        pass

    def __call__(self, X):
        self.X = X
        return np.where(X > 0, X, 0)

    def backward(self, out):
        return np.where(self.X > 0, 1, 0) * out

    def parameters(self):
        return None


class Sequential:
    def __init__(self, *args):
        self.layers = args

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, out):
        for layer in reversed(self.layers):
            out = layer.backward(out)
        return out

    def parameters(self):
        for layer in self.layers:
            params = layer.parameters()
            if params is not None:
                for param in params:
                    yield param
