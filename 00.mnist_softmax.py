import tensorflow as tf
import numpy as np
from tensorflow import keras
mnist = keras.datasets.mnist

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images/255., test_images/255.

# Training parameters
training_epochs = 30


