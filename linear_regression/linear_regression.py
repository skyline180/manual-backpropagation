import numpy as np

X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([[2], [4], [6], [8]], dtype=float)

w = np.random.randn(1, 1)
b = np.zeros((1,))

lr = 0.01

for epoch in range(1000):

    y_pred = X @ w + b

    loss = np.mean((y_pred - y) ** 2)

    grad_y_pred = 2 * (y_pred - y) / y.shape[0]

    grad_w = X.T @ grad_y_pred
    grad_b = np.sum(grad_y_pred)

    w -= lr * grad_w
    b -= lr * grad_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print("Final weight:", w)
print("Final bias:", b)