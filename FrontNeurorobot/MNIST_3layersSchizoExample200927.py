#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import schizo

(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
#keras.backend.image_data_format()
# 'channels_last'
training_images, test_images = training_images / 255.0, test_images / 255.0
training_labels = tf.keras.utils.to_categorical(training_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

model_normal = keras.Sequential([
  layers.Flatten(),
  layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
  layers.Dense(10, activation='softmax')
])
model_normal.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('****Normal network****')
results = model_normal.fit(training_images, training_labels, batch_size=32, epochs=10, validation_data=(test_images, test_labels))

model = keras.Sequential([
  layers.Flatten(),
  schizo.SzDense(512, param_reduction=0.7, form='diagonal', activation='relu', kernel_initializer='he_normal'),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('****Schizophrenia network****')
results = model.fit(training_images, training_labels, batch_size=32, epochs=10, validation_data=(test_images, test_labels))

print('actual param reduction=', model.layers[1].get_reduced_ratio())
print('half band width=', model.layers[1].get_halfbandwidth())
print('weight=')
print(model.layers[1].get_weights()[0])
print('window=')
print(model.layers[1].get_weights()[2])

