import numpy as np

class MaxPool2D:

    def forward(self, image):

        self.image = image

        h, w = image.shape

        output = np.zeros((h//2, w//2))

        for i in range(0, h, 2):
            for j in range(0, w, 2):

                region = image[i:i+2, j:j+2]

                output[i//2, j//2] = np.max(region)

        return output