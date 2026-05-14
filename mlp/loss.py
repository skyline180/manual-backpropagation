import numpy as np

class MSELoss:

    def forward(self, pred, target):

        return np.mean((pred - target) ** 2)

    def backward(self, pred, target):

        return 2 * (pred - target) / target.shape[0]