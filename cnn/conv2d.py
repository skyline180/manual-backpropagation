import numpy as np

class Conv2D:

    def __init__(self, kernel_size):

        self.kernel = np.random.randn(kernel_size, kernel_size) * 0.1

    def forward(self, image):

        self.image = image

        h, w = image.shape
        k = self.kernel.shape[0]

        output = np.zeros((h-k+1, w-k+1))

        for i in range(h-k+1):
            for j in range(w-k+1):

                region = image[i:i+k, j:j+k]

                output[i, j] = np.sum(region * self.kernel)

        return output

    def backward(self, grad_output, lr):

        k = self.kernel.shape[0]

        grad_kernel = np.zeros_like(self.kernel)

        for i in range(grad_output.shape[0]):
            for j in range(grad_output.shape[1]):

                region = self.image[i:i+k, j:j+k]

                grad_kernel += grad_output[i, j] * region

        self.kernel -= lr * grad_kernel