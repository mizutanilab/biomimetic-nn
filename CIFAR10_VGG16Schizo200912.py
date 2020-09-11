#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import math
import numpy as np
from numpy.random import seed
import random
import time
import gc
from schizo import Schizo

stitle = 'CIFAR10_VGG16Schizo200912'
num_repeat = 2
num_epoch = 1
idim = 512
randomseed = 5
num_batch = 100
output_mode = 1
#output_mode = 0: no console output (minimum output)
#output_mode = 1: show progress bar (jupyter notebook)
#output_mode = 2: one line per epoch (shell script)

idlist = [0.5]
len_idlist = len(idlist)
evalout = [[0] * 4 for i in range(len_idlist)]

def Learning_rate_by_step(epoch):
  lrnRate = 0.0005
  if(epoch >= 120):
    lrnRate /= 5
  return lrnRate

seed(randomseed)
tf.random.set_seed(randomseed)
random.seed(a=randomseed, version=2)

from tensorflow.keras.datasets import cifar10
num_class = 10
(training_images, training_labels), (test_images, test_labels) = cifar10.load_data()
keras.backend.image_data_format()
# 'channels_last'
training_images, test_images = training_images / 255.0, test_images / 255.0
training_labels = tf.keras.utils.to_categorical(training_labels, num_class)
test_labels = tf.keras.utils.to_categorical(test_labels, num_class)
training_images = training_images.astype('float32')
test_images = test_images.astype('float32')
training_steps = training_images.shape[0] // num_batch
validation_steps = test_images.shape[0] // num_batch

path = stitle + '.txt'
flog = open(path, mode='w')
flog.close()

start = time.time()
ilist = 0
for id in idlist:
  fcount=0
  irepeat = 0
  for irpt in range(num_repeat):
    # schizophrenia mimicking layer: (ilayer+1)-th layer
    ilayer = -26
    if output_mode != 0:
      print('id=', id, 'irepeat=', irepeat, '/', num_repeat)
    model = keras.Sequential([
      layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=training_images.shape[1:], activation='relu', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      #layers.Activation('relu'),
      layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'), 
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      #
      layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      #
      layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(),
      #
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(),
      #
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      #
      #layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
      #layers.Dropout(0.5),
      #layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
      #layers.Dropout(0.5),
      #layers.Dense(1024, activation='relu', kernel_initializer='he_normal'),
      #layers.Dropout(0.5),
      Schizo(4096, reduction_ratio=id, form='diagonal', activation='relu', kernel_initializer='he_normal'),
      Schizo(4096, reduction_ratio=id, form='diagonal', activation='relu', kernel_initializer='he_normal'),
      Schizo(1024, reduction_ratio=id, form='diagonal', activation='relu', kernel_initializer='he_normal'),
      layers.Dense(num_class, activation='softmax')
    ])
    #opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    results = model.fit(ImageDataGenerator(rotation_range = 20, horizontal_flip = True, height_shift_range = 0.2, 
                              width_shift_range = 0.2,zoom_range = 0.2, channel_shift_range = 0.2
                             ).flow(training_images, training_labels, num_batch), 
                        batch_size=num_batch, epochs=num_epoch, steps_per_epoch = training_steps, 
                        validation_data=ImageDataGenerator().flow(test_images, test_labels, num_batch), 
                        validation_steps = validation_steps, 
                        callbacks = [LearningRateScheduler(Learning_rate_by_step)], 
                        verbose=output_mode)
    ###file output
    with open(path, mode='a') as flog:
      print('irpt/num_repeat=', irepeat, '/', num_repeat, file=flog)
      keys = results.history.keys()
      len_results = len(results.history[list(keys)[0]])
      vals = ''
      for k in keys:
        vals += '\t' + k
      print('step', vals, file=flog)
      for istep in range(len_results):
        vals = ''
        for k in keys:
          vals += '\t' + str(results.history[k][istep])
        print(istep+1, vals, file=flog)
    ###file output
    scores = model.evaluate(test_images, test_labels, verbose=output_mode)
    evalout[ilist][0] = id
    if ilayer >= 0:
      evalout[ilist][0] = model.layers[ilayer].get_halfwidth()
      evalout[ilist][1] = model.layers[ilayer].get_reduced_ratio()
    evalout[ilist][2] += scores[0]
    evalout[ilist][3] += scores[1]
    irepeat += 1
    keras.backend.clear_session()
    del model
    gc.collect()
  ###for irpt
  if irepeat > 0:
    evalout[ilist][2] /= irepeat
    evalout[ilist][3] /= irepeat
  ###if
  with open(path, mode='a') as flog:
    print('width=', evalout[ilist][0], 'reduced=', evalout[ilist][1], 'loss=', evalout[ilist][2], 'acc=', evalout[ilist][3], file=flog)
  ilist += 1
#for id
if output_mode != 0:
  elapsed_time = time.time() - start
  print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
  print('*****finished*****')
#endif

with open(path, mode='a') as flog:
  print(stitle, file=flog)
  print('seed\t', 'dim\t', 'epoch\t', 'repeat', file=flog)
  print(randomseed, '\t', idim, '\t', num_epoch, '\t', num_repeat, file=flog)
  print('width\t','reduction\t','loss\t','accuracy', file=flog)
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2], '\t', evalout[ie][3], file=flog)
  #for ie
#open

if output_mode != 0:
  print(stitle)
  print('seed\t', 'dim\t', 'epoch\t', 'repeat')
  print(randomseed, '\t', idim, '\t', num_epoch, '\t', num_repeat)
  print('width\t','reduction\t','loss\t','accuracy')
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2], '\t', evalout[ie][3])
  #for ie
#endif



