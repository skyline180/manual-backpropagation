import numpy as np

from dense import Dense
from activations import Sigmoid
from loss import MSELoss

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

dense1 = Dense(2, 4)
act1 = Sigmoid()

dense2 = Dense(4, 1)
act2 = Sigmoid()

loss_fn = MSELoss()

lr = 0.1

for epoch in range(10000):

    out1 = dense1.forward(X)
    out1 = act1.forward(out1)

    out2 = dense2.forward(out1)
    pred = act2.forward(out2)

    loss = loss_fn.forward(pred, y)

    grad = loss_fn.backward(pred, y)

    grad = act2.backward(grad)
    grad = dense2.backward(grad, lr)

    grad = act1.backward(grad)
    grad = dense1.backward(grad, lr)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print(pred)