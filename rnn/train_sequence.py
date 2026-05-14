import numpy as np

from rnn_cell import RNNCell

rnn = RNNCell(input_size=1, hidden_size=8)

sequence = [
    np.array([1]),
    np.array([2]),
    np.array([3]),
    np.array([4]),
]

outputs = rnn.forward(sequence)

for out in outputs:
    print(out)