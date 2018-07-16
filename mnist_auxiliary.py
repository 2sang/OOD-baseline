from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import experiments
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy

def train():
    
    ### TRAIN MODEL
    training_epochs = 10
    image_size = 28
    input_dim = image_size * image_size
    n_labels = 10
    bottleneck_dim = 128
    learning_rate = 0.001
    batch_size = 128
    
    # Base model
    inputs = Input(shape=(input_dim, ), name='input')
    h1 = Dense(256, activation='relu')(inputs)
    h2 = Dense(256, activation='relu')(h1)
    
    # Softmax logits output
    h3 = Dense(256, activation='relu')(h2)
    logits_out = Dense(n_labels, activation='softmax', name='logits_output')(h3)
    
    # Reconstruction image output
    bottleneck = Dense(bottleneck_dim, activation='relu')(h2)
    decode1 = Dense(256, activation='relu')(bottleneck)
    decode2 = Dense(256, activation='relu')(decode1)
    reconstruction = Dense(input_dim, name='rec_output')(decode2)
    
    # Instantiate base model
    base_model = Model(inputs, [h2, logits_out, reconstruction], name='base')
    
    base_model.compile(optimizer='adam', 
                       loss={'logits_output': 'sparse_categorical_crossentropy',
                             'rec_output': 'mean_squared_error'},
                       loss_weights={'logits_output': 0.9,
                                     'rec_output': 0.1})
            
    base_model.fit(mnist_train_x,
                   {'logits_output': mnist_train_y,
                    'rec_output': mnist_train_x},
                   epochs=training_epochs, batch_size=batch_size)
    
    #test_loss, test_acc = base_model.evaluate(mnist_test_x, mnist_test_y)
    #print("Training done, test accuracy: {}".format(test_acc))
    
    ### SAVE MODEL
    keras.models.save_model(base_model, "./mnist_aux_base.hdf5")
    return base_model

# Load MNIST, FMNIST dataset
mnist = keras.datasets.mnist
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()
mnist_train_x, mnist_test_x = np.reshape(mnist_train_x, [-1, 28*28]), np.reshape(mnist_test_x, [-1, 28*28])
mnist_train_x, mnist_test_x = mnist_train_x/255., mnist_test_x/255.

fashion_mnist = keras.datasets.fashion_mnist
(fmnist_train_x, fmnist_train_y), (fmnist_test_x, fmnist_test_y) = fashion_mnist.load_data()
fmnist_train_x, fmnist_test_x = fmnist_train_x/255., fmnist_test_x/255.

saved_model_path = './mnist_ax.hdf5'
if not os.path.exists(saved_model_path):
    model = train()
else:
    model = keras.models.load_model(saved_model_path)
    
# experiments.right_wrong_distinction(model, mnist_test_x, mnist_test_y)
# experiments.in_out_distribution_distinction(model, mnist_test_x, fmnist_test_x, "FashionMNIST")
# experiments.in_out_distribution_distinction(model, mnist_test_x, np.random.normal(size=(10000, 28, 28)), "WhiteNoise")
# experiments.in_out_distribution_distinction(model, mnist_test_x, np.random.uniform(size=(10000, 28, 28)), "UniformNoise")
