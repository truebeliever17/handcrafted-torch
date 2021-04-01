import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true, eps=1e-32):
        y_pred -= np.max(y_pred) 
        exp_sum = np.sum(np.exp(y_pred), axis=1)
        x = np.choose(y_true, y_pred.T)
        loss = np.mean(-x + np.log(exp_sum + eps))
        grad = np.exp(y_pred) / (exp_sum + eps).reshape(-1, 1)
        grad[np.arange(grad.shape[0]), y_true] -= 1
        return loss, grad
