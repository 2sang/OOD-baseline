import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images/255., test_images/255.

if not os.path.exists("./mnist.hdf5"):
    train()
else:
    load_model('mnist')
    
run_experiment()


def train():
    ### TRAINING STAGE
    training_epochs = 3
    learning_rate = 0.001
    batch_size = 128

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=training_epochs, batch_size=batch_size)

    ### TEST STAGE
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy: {}".format(test_acc))
    
    ### SAVE MODEL
    keras.models.save_model(model, "mnist")
    

predictions = model.predict(test_images)

def run_experiment():
    pass
