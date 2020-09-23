#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import math
import numpy as np
from numpy.random import seed
import random
import time
import gc
import schizo

stitle = 'MNIST_3layersSchizo200905'
num_repeat = 100
num_epoch = 50
idim = 512
randomseed = 1
num_batch = 32
output_mode = 2
#output_mode = 0: no console output (minimum output)
#output_mode = 1: show progress bar (jupyter notebook)
#output_mode = 2: one line per epoch (shell script)

idlist =[0.0, 0.167, 0.333, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925]
len_idlist = len(idlist)
evalout = [[0] * 4 for i in range(len_idlist)]

seed(randomseed)
tf.random.set_seed(randomseed)
random.seed(a=randomseed, version=2)

num_class = 10
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
#keras.backend.image_data_format()
# 'channels_last'
training_images, test_images = training_images / 255.0, test_images / 255.0
training_labels = tf.keras.utils.to_categorical(training_labels, num_class)
test_labels = tf.keras.utils.to_categorical(test_labels, num_class)

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
    ilayer = 1
    if output_mode != 0:
      print('id=', id, 'irepeat=', irepeat, '/', num_repeat)
    model = keras.Sequential([
      layers.Flatten(),
      schizo.SzDense(idim, param_reduction=id, form='diagonal', activation='relu', kernel_initializer='he_normal'),
      layers.Dense(num_class, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    results = model.fit(training_images, training_labels, batch_size=num_batch, epochs=num_epoch, 
                        validation_data=(test_images, test_labels), shuffle=True, verbose=output_mode)
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
    print('width=', evalout[ilist][0], 'frac=', evalout[ilist][1], 'loss=', evalout[ilist][2], 'acc=', evalout[ilist][3], file=flog)
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
  print('width\t','frac\t','loss\t','accuracy', file=flog)
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2], '\t', evalout[ie][3], file=flog)
  #for ie
#open

if output_mode != 0:
  print(stitle)
  print('seed\t', 'dim\t', 'epoch\t', 'repeat')
  print(randomseed, '\t', idim, '\t', num_epoch, '\t', num_repeat)
  print('width\t','frac\t','loss\t','accuracy')
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2], '\t', evalout[ie][3])
  #for ie
#endif

