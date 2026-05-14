Backpropagation is an algorithm used to compute gradients of the loss function with respect to neural network weights.
It works using the chain rule from calculus and allows neural networks to learn by updating weights to reduce prediction error.

---

**The goal is to minimize the loss function.**

Weight update rule:

w = w - learning_rate \* gradient

## Forward Pass

Steps in the forward pass:

1. Input enters the network
2. Compute weighted sums
3. Apply activation functions
4. Generate predictions
5. Compute loss

Example:

z = w\*x + b

a = sigmoid(z)

## Loss Function

Example Mean Squared Error (MSE):

L = (y_pred - y_true)^2

The loss tells us how wrong the prediction is.

## Backward Pass

The backward pass computes gradients from output layer back to earlier layers.

Main idea:

- Find how much each weight contributed to the error
- Update weights to reduce future error

## Chain Rule

Backpropagation uses the chain rule.

Example:

dL/dw = dL/dy _ dy/dz _ dz/dw

This allows gradients to flow backward through the network.

## Sigmoid Activation

Sigmoid function:

sigmoid(x) = 1 / (1 + e^-x)

Derivative:

sigmoid'(x) = sigmoid(x) \* (1 - sigmoid(x))

## Gradient Descent

Weights are updated using gradients:

w = w - lr \* gradient

Where:

- w = weight
- lr = learning rate

## References

- Neural Networks and Deep Learning
- Calculus chain rule
- Gradient descent optimization
- PyTorch autograd system
