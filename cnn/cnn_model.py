import numpy as np

from conv2d import Conv2D
from maxpool import MaxPool2D

image = np.random.randn(28, 28)

conv = Conv2D(3)
pool = MaxPool2D()

out = conv.forward(image)

print("Conv output shape:", out.shape)

out = pool.forward(out)

print("Pool output shape:", out.shape)