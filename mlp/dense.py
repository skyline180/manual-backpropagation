import numpy as np

class Dense:

    def __init__(self, input_size, output_size):

        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))

    def forward(self, x):

        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output, lr):

        grad_W = np.dot(self.x.T, grad_output)
        grad_b = np.sum(grad_output, axis=0, keepdims=True)

        grad_input = np.dot(grad_output, self.W.T)

        self.W -= lr * grad_W
        self.b -= lr * grad_b

        return grad_input