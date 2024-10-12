#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.constraints import min_max_norm
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
import math
import random
import numpy
from numpy.random import seed
import time

stitle = 'MNIST_4layersStripingNrm200825'
num_repeat = 2
num_epoch = 20
idim = 512
randomseed = 1
num_batch = 32
##commet-in if normalize weights

#idlist: number of stripes
im = int(idim / 100)
if im < 1:
  im = 1
idlist = list(range(0, idim, im))

len_idlist = len(idlist)
evalout = [[0] * 4 for i in range(len_idlist)]

seed(randomseed)
tf.random.set_seed(randomseed)
random.seed(a=randomseed, version=2)

num_class = 10
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images / 255.0
training_labels = tf.keras.utils.to_categorical(training_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

start = time.time()
irepeat = 0
for irpt in range(num_repeat):
  # schizophrenia mimicking layer: (ilayer+1)-th layer
  ilayer = 2
  model = tf.keras.models.Sequential([  
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(idim, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(idim, activation='relu', kernel_initializer='he_normal', use_bias=False),
    tf.keras.layers.Dense(num_class, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(training_images, training_labels, batch_size=num_batch, epochs=num_epoch, verbose=1)
  # model.summary()
  # model.evaluate(test_images, test_labels)
  lscz = model.layers[ilayer]
  lsczwstore = lscz.get_weights()
  # tf.shape(lscz.get_weights()[0])
  # print(lscz.get_weights()[1])
  # for w in model.layers[ilayer].weights:
  #   print('{:<25}{}'.format(w.name, w.shape))
  print('irpt/num_repeat=', irpt, '/', num_repeat)
  ilist = 0
  # random.seed(a=randomseed, version=2)
  for id in idlist:
    fcount=0
    if id > idim:
      continue
    if id > 0:
      lsczw = lscz.get_weights()
      ######
      fstep = idim / id
      for ia in range(id):
        iy0 = int(fstep * ia)
        for ix in range(idim):
          iy = (ix + iy0) % idim
          lsczw[0][ix][iy] = 0
          fcount += 1
        #for ix end
      ######
      fcount = 1. - fcount / (idim * idim)
      nplsczw = numpy.array(lsczw)
      #normalization
      if fcount > 0:
        nplsczw = nplsczw / fcount
      model.layers[ilayer].set_weights(nplsczw)
      fcount = 1. - fcount
    #if id>0
    # print('id=', id, 'count=', fcount)
    scores = model.evaluate(test_images, test_labels, verbose=0)
    evalout[ilist][0] = id
    evalout[ilist][1] = fcount
    evalout[ilist][2] += scores[0]
    evalout[ilist][3] += scores[1]
    ilist += 1
    model.layers[ilayer].set_weights(lsczwstore)
    if (ilist % 10) == 0:
      ltw = model.layers[ilayer].get_weights()
      zcount = 0
      for ix in range(idim):
        for iy in range(idim):
          if (ltw[0][ix][iy] == 0):
            zcount += 1
      print ('id=', id, 'count=', fcount, 'zcount=', zcount)
  #for id
  irepeat += 1
#for irpt
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print('*****finished*****')

path = stitle + '.txt'
with open(path, mode='w') as f:
  print(stitle, file=f)
  print('seed\t', 'dim\t', 'epoch\t', 'repeat', file=f)
  print(randomseed, '\t', idim, '\t', num_epoch, '\t', irepeat, file=f)
  print('id\t','fraction\t','loss\t','accuracy', file=f)
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2] / irepeat, '\t', evalout[ie][3] / irepeat, file=f)
  #for ie
#open

print(stitle)
print('seed\t', 'dim\t', 'epoch\t', 'repeat')
print(randomseed, '\t', idim, '\t', num_epoch, '\t', irepeat)
print('id\t','fraction\t','loss\t','accuracy')
for ie in range(len_idlist):
  print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2] / irepeat, '\t', evalout[ie][3] / irepeat)
#for ie
