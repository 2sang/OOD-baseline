import os
import tensorflow as tf
import numpy as np
import experiments
from tensorflow import keras

# Define train(). execution starts from the bottom
def train():
    ### TRAIN MODEL
    training_epochs = 10
    learning_rate = 0.001
    batch_size = 128

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(mnist_train_x, mnist_train_y, epochs=training_epochs, batch_size=batch_size)

    test_loss, test_acc = model.evaluate(mnist_test_x, mnist_test_y)
    print("Training done, test accuracy: {}".format(test_acc))
    
    ### SAVE MODEL
    keras.models.save_model(model, "./mnist.hdf5")
    return model


# Load MNIST, FMNIST dataset
mnist = keras.datasets.mnist
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()
mnist_train_x, mnist_test_x = mnist_train_x/255., mnist_test_x/255.

fashion_mnist = keras.datasets.fashion_mnist
(fmnist_train_x, fmnist_train_y), (fmnist_test_x, fmnist_test_y) = fashion_mnist.load_data()
fmnist_train_x, fmnist_test_x = fmnist_train_x/255., fmnist_test_x/255.

saved_model_path = './mnist.hdf5'
if not os.path.exists(saved_model_path):
    model = train()
else:
    model = keras.models.load_model(saved_model_path)
    
experiments.right_wrong_distinction(model, mnist_test_x, mnist_test_y)
experiments.in_out_distribution_distinction(model, mnist_test_x, fmnist_test_x, "FashionMNIST")
print("fmnist_test_x.shape: {}".format(fmnist_test_x.shape))
experiments.in_out_distribution_distinction(model, mnist_test_x, np.random.normal(size=(10000, 28, 28)), "WhiteNoise")
experiments.in_out_distribution_distinction(model, mnist_test_x, np.random.uniform(size=(10000, 28, 28)), "UniformNoise")
