import numpy as np
from skimage.filters import gaussian

distortion = np.random.uniform(low=0.9, high=1.2)

def add_noise(batch, complexity=0.5):
    return batch + np.random.normal(size=batch.shape, scale=1e-9 + complexity)

def add_distortion_noise(batch):
    return batch + np.random.normal(size=batch.shape, scale=1e-9 + distortion)

def add_distortion_blur(img):
    image = img.reshape((-1, 28, 28))
    return gaussian(image, sigma=5*distortion).reshape((-1, 28*28))

def rotate90_if_not_zero(batch, batch_label):
    mask = batch_label != 0
    
    nonzeros = batch[mask]
    nonzeros = np.rot90(nonzeros.reshape((-1, 28, 28))).reshape((-1, 28*28))
    zeros = batch[np.invert(mask)]
    zeros = add_noise(zeros, complexity=1)
    batch = np.vstack((nonzeros, zeros))
    
    return batch

