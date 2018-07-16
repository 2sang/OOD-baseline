import numpy as np
from skimage.filters import gaussian

def add_noise(batch, complexity=0.5):
    return batch + np.random.normal(size=batch.shape, scale=1e-9 + complexity)

def blur(img, complexity=0.5):
    image = img.reshape((-1, 28, 28))
    return gaussian(image, sigma=5*complexity).reshape((-1, 28*28))
