import numpy as np

class Sigmoid:

    def forward(self, x):

        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad_output):

        return grad_output * self.out * (1 - self.out)


class ReLU:

    def forward(self, x):

        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):

        grad = grad_output.copy()
        grad[self.x <= 0] = 0
        return grad