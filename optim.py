import numpy as np


class SGD:
    def __init__(self, parameters, lr, momentum=None):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum

    def step(self):
        for param in self.parameters:
            if param is not None:
                if self.momentum is None:
                    param.data -= self.lr * param.grad
                    continue          

                param.velocity = self.momentum * param.velocity + self.lr * param.grad
                param.data -= param.velocity

    def zero_grad(self):
        for param in self.parameters:
            if param is not None:
                param.grad = np.zeros_like(param.data)
