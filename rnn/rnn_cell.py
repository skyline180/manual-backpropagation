import numpy as np

class RNNCell:

    def __init__(self, input_size, hidden_size):

        self.hidden_size = hidden_size

        self.Wxh = np.random.randn(input_size, hidden_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(hidden_size, 1) * 0.1

        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, 1))

    def forward(self, inputs):

        h = np.zeros((1, self.hidden_size))

        self.hs = []
        self.outputs = []

        for x in inputs:

            x = x.reshape(1, -1)

            h = np.tanh(
                x @ self.Wxh +
                h @ self.Whh +
                self.bh
            )

            y = h @ self.Why + self.by

            self.hs.append(h)
            self.outputs.append(y)

        return self.outputs