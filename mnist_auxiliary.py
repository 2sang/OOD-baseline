from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import experiments
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy


def train_base():
    
    ### TRAIN MODEL
    training_epochs = 10
    image_size = 28
    input_dim = image_size * image_size
    n_labels = 10
    bottleneck_dim = 128
    merge_layer_dim = 512
    learning_rate = 0.001
    batch_size = 128
    
    # Base model
    inputs = Input(shape=(input_dim, ), name='image_input')
    h1 = Dense(256, activation='relu', name='h1')(inputs)
    h2 = Dense(256, activation='relu', name='h2')(h1)
    
    # Softmax logits output
    h3 = Dense(256, activation='relu', name='h3')(h2)
    logits_out = Dense(n_labels, activation='softmax', name='logits_output')(h3)
    
    # Reconstruction image output
    bottleneck = Dense(bottleneck_dim, activation='relu', name='bottleneck')(h2)
    decode1 = Dense(256, activation='relu', name='decode1')(bottleneck)
    decode2 = Dense(256, activation='relu', name='decode2')(decode1)
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
    
    #test_loss, test_acc 
    test_result = base_model.evaluate(x=mnist_test_x,
                                      y={'logits_output': mnist_test_y,
                                         'rec_output': mnist_test_x})

    print("metric names:", base_model.metrics_names)
    print(test_result)
    
    ### SAVE MODEL
    keras.models.save_model(base_model, "./mnist_aux_base.hdf5")
    return base_model


class Merge3Ways(keras.layers.Layer):
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Merge3Ways, self).__init__(**kwargs)
        
    def build(self, input_shape):
        shape_x, shape_h2, shape_logits, shape_rec =\
                list(map(lambda shape: int(shape[1]), input_shape))
        
        self.weight1 = self.add_weight(name='h2_to_merge',
                                       shape=(shape_h2, self.output_dim))
        self.weight2 = self.add_weight(name='logits_to_merge',
                                       shape=(shape_logits, self.output_dim))
        self.weight3 = self.add_weight(name='rec_to_merge',
                                       shape=(shape_rec, self.output_dim))
        super(Merge3Ways, self).build(input_shape)
        
        
    def call(self, inputs):
        x, h2, logits_out, reconstruction = inputs
        a1 = K.dot(h2, self.weight1)
        a2 = K.dot(logits_out, self.weight2)
        a3 = K.dot(K.square(reconstruction-x), self.weight3)
        return a1 + a2 + a3
    

# Continue to base model, merge all output layers to external module
def train_abnormal_model(base_model):
    
    merge_layer_dim = 512
    input_dim = 28*28
        
    # Need to point index 0.
    image_inputs = base_model.inputs[0]
    
    # Deconstruct base model outputs,
    h2, logits_out, reconstruction = base_model.outputs
    
    # Merging layer
    merged = Merge3Ways(512)([image_inputs, h2, logits_out, reconstruction])
    risk_1 = Dense(128, activation='relu', name='risk_1')(merged)
    risk_out = Dense(1, activation='relu', name='risk_out')(risk_1)
    
    
    aux_model = Model(image_inputs, risk_out)
    base_model.summary()
    aux_model.summary()
    
    # Freeze base model layers 
    

# Load MNIST, FMNIST dataset
mnist = keras.datasets.mnist
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()
mnist_train_x, mnist_test_x = np.reshape(mnist_train_x, [-1, 28*28]), np.reshape(mnist_test_x, [-1, 28*28])
mnist_train_x, mnist_test_x = mnist_train_x/255., mnist_test_x/255.

fashion_mnist = keras.datasets.fashion_mnist
(fmnist_train_x, fmnist_train_y), (fmnist_test_x, fmnist_test_y) = fashion_mnist.load_data()
fmnist_train_x, fmnist_test_x = np.reshape(fmnist_train_x, [-1, 28*28]), np.reshape(fmnist_test_x, [-1, 28*28])
fmnist_train_x, fmnist_test_x = fmnist_train_x/255., fmnist_test_x/255.


base_model_path = './mnist_aux_base.hdf5'
abnormal_model_path = './mnist_abnormal.hdf5'

if not os.path.exists(base_model_path):
    base_model = train_base()
else:
    base_model = keras.models.load_model(base_model_path)
    
if not os.path.exists(abnormal_model_path):
    abnormal_model = train_abnormal_model(base_model)
else:
    abnormal_model = keras.models.load_model(abnormal_model_path)
    
# experiments.right_wrong_distinction(model, mnist_test_x, mnist_test_y)
# experiments.in_out_distribution_distinction(model, mnist_test_x, fmnist_test_x, "FashionMNIST")
# experiments.in_out_distribution_distinction(model, mnist_test_x, np.random.normal(size=(10000, 28, 28)), "WhiteNoise")
# experiments.in_out_distribution_distinction(model, mnist_test_x, np.random.uniform(size=(10000, 28, 28)), "UniformNoise")
