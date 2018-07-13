import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define train() & run_experiment(). execution starts from line 000
def train():
    ### TRAIN MODEL
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

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy: {}".format(test_acc))
    
    ### SAVE MODEL
    keras.models.save_model(model, "./mnist.hdf5")
    return model

def run_experiment(model):
    predictions = model.predict(test_images)
    print("EXPERIMENT")
    
saved_model_path = './mnist.hdf5'
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images/255., test_images/255.
    
    
if not os.path.exists(saved_model_path):
    model = train()
else:
    model = keras.models.load_model(saved_model_path)
run_experiment(model)
