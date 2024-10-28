#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
import math
import random
import numpy
from numpy.random import seed
import time
import gc

stitle = 'CIFAR10_DiagNrmCNN200909'
num_repeat = 30
num_epoch = 100
idim = 512
randomseed = 1
num_batch = 32
output_mode = 2
#output_mode = 1: show progress bar (jupyter notebook)
#output_mode = 0: formatted output (python3 command line)

# idlist = [idim]
#im = int(idim / 50)
#if im < 1:
#  im = 1
idlist = list(range(0, 100, 1))

len_idlist = len(idlist)
evalout = [[0] * 4 for i in range(len_idlist)]

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

path = stitle + '.txt'
flog = open(path, mode='w')
flog.close()

start = time.time()
irepeat = 0
for irpt in range(num_repeat):
  # schizophrenia mimicking layer: (ilayer+1)-th layer
  ilayer = 9
  if output_mode != 0:
    print('irepeat=', irepeat, '/', num_repeat)
  model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same', input_shape=training_images.shape[1:]),
    keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(idim, activation='relu', kernel_initializer='he_normal', use_bias=False),
    keras.layers.Dense(num_class, activation='softmax')
  ])
  opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
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
  lscz = model.layers[ilayer]
  lsczwstore = lscz.get_weights()
  ilist = 0
  for id in idlist:
    fcount=0
    #if id > idim:
    #  continue
    lsczw = lscz.get_weights()
    nx = model.layers[ilayer-1].output.shape[1]
    ny = idim
    rxy = (nx-1) / (ny-1)
    hw = (nx*ny / math.sqrt(nx*nx + ny*ny)) * (1. - math.sqrt(id * 0.01))
    hwdiv = hw * math.sqrt(rxy * rxy + 1)
    ######
    for iy in range(ny):
      for ix in range(nx):
        if (abs(rxy * iy - ix) >= hwdiv):
          lsczw[0][ix][iy] = 0
        else:
          fcount += 1
        #end if
      #for ix end
      #  lsczw[1][iy] = 0
    ######
    fcount = fcount / (nx * ny)
    nplsczw = numpy.array(lsczw)
    if fcount > 0:
        nplsczw = nplsczw / fcount
    model.layers[ilayer].set_weights(nplsczw)
    fcount = 1. - fcount
    scores = model.evaluate(test_images, test_labels, verbose=0)
    evalout[ilist][0] = hw
    evalout[ilist][1] = fcount
    evalout[ilist][2] += scores[0]
    evalout[ilist][3] += scores[1]
    with open(path, mode='a') as flog:
      print('hw=', evalout[ilist][0], 'frac=', evalout[ilist][1], 'loss=', scores[0], 'acc=', scores[1], file=flog)
    ilist += 1
    model.layers[ilayer].set_weights(lsczwstore)
    if ((ilist % 10) == 0) and (output_mode != 0):
      ltw = model.layers[ilayer].get_weights()
      zcount = 0
      for ix in range(nx):
        for iy in range(ny):
          if (ltw[0][ix][iy] == 0):
            zcount += 1
      print ('id=', id, 'count=', fcount, 'zcount=', zcount)
  #loop end
  irepeat += 1
  del lscz
  del lsczw
  del lsczwstore
  del nplsczw
  del ltw
  gc.collect()
  keras.backend.clear_session()
#repetition loop ends
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print('*****finished*****')

with open(path, mode='a') as f:
  print(stitle, file=f)
  print('seed\t', 'dim\t', 'epoch\t', 'repeat', file=f)
  print(randomseed, '\t', idim, '\t', num_epoch, '\t', irepeat, file=f)
  print('id\t','fraction\t','loss\t','accuracy', file=f)
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2] / irepeat, '\t', evalout[ie][3] / irepeat, file=f)
  #for ie
#open

if output_mode != 0:
  print(stitle)
  print('seed\t', 'dim\t', 'epoch\t', 'repeat')
  print(randomseed, '\t', idim, '\t', num_epoch, '\t', irepeat)
  print('id\t','fraction\t','loss\t','accuracy')
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2] / irepeat, '\t', evalout[ie][3] / irepeat)
  #for ie
#endif


