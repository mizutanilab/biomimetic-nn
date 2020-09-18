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

stitle = 'CIFAR10_CNNSchizo200910'
num_repeat = 10
num_epoch = 100
idim = 512
randomseed = 1
num_batch = 32
output_mode = 2
#output_mode = 0: no console output (minimum output)
#output_mode = 1: show progress bar (jupyter notebook)
#output_mode = 2: one line per epoch (shell script)

idlist = [0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95]
len_idlist = len(idlist)
evalout = [[0] * 4 for i in range(len_idlist)]

class Schizo(layers.Layer):
  def __init__(self, output_dim, halfwidth=0, reduction_ratio=0, form='diagonal', activation='relu', kernel_initializer='he_normal', **kwargs):
    self.output_dim = output_dim
    self.halfwidth = halfwidth
    self.form = form
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.reduction_sv = reduction_ratio
    super(Schizo, self).__init__(**kwargs)
  def build(self, input_shape):
    # assert K.image_data_format() == "channels_last"
    self.input_xdim = input_shape[1]
    self.kernel = self.add_weight(name='kernel',
                                  shape=(input_shape[1], self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
    self.window = self.add_weight(name='window',
                                  shape=(input_shape[1], self.output_dim),
                                  initializer='ones',
                                  trainable=False)
    self.bias   = self.add_weight(name='bias',
                                  shape=(1, self.output_dim),
                                  initializer='zeros',
                                  trainable=True)
    self.num_ones = 0
    self.reduced_ratio = 0
    nx = self.input_xdim
    ny = self.output_dim
    self.num_weights = nx * ny
    if self.halfwidth == 0:
      self.halfwidth = (nx*ny / math.sqrt(nx*nx + ny*ny)) * (1. - math.sqrt(self.reduction_sv)) 
      if self.form == 'gaussian':
        self.halfwidth *= 1.5
    #endif
    #wnd = [[0] * ny for i in range(nx)]
    wnd = np.zeros((nx,ny))
    w_corr = 1.
    if self.form == 'diagonal':
      if ny > 1:
        rxy = (nx-1) / (ny-1)
        hwdiv = self.halfwidth * math.sqrt(rxy * rxy + 1)
        for iy in range(ny):
          ix1 = rxy * iy - hwdiv
          ix1 = int(ix1) + 1 if ix1 >= 0 else 0
          if ix1 > nx-1:
            continue
          ix2 = rxy * iy + hwdiv
          ix2 = math.ceil(ix2) if ix2 < nx else nx
          wnd[ix1:ix2, iy:iy+1] = 1
          self.num_ones += (ix2-ix1)
          #for ix in range(ix1, ix2):
          #  wnd[ix][iy] = 1
          #  self.num_ones += 1
        #for ixiy
      else:
        wnd[:,:] = 1
        self.num_ones += nx
        #for ix in range(nx):
        #  wnd[ix][0] = 1
        #  self.num_ones += 1
      #endif ny>1
      self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
      if self.num_ones > 0:
        w_corr = self.num_weights / self.num_ones
      self.kernel.assign(self.kernel * (wnd * w_corr))
    elif self.form == 'gaussian':
      if (self.halfwidth > 0) and (ny > 1):
        sgm2 = 1. / (2. * self.halfwidth * self.halfwidth)
        gsum = 0
        rxy = (nx-1) / (ny-1)
        for ix in range(nx):
          for iy in range(ny):
            gauss = math.exp(-(ix-rxy*iy)*(ix-rxy*iy)*sgm2)
            wnd[ix][iy] = gauss
            gsum += gauss
        #for ixiy
        self.reduced_ratio = 1. - gsum / self.num_weights
        if gsum > 0:
          w_corr = self.num_weights / gsum
        wnd = wnd * w_corr
      else:
        wnd[:,:] = 1
        self.num_ones = nx * ny
        #for ix in range(nx):
        #  for iy in range(ny):
        #    wnd[ix][iy] = 1
      #endif halfwidth
    #endif form_function
    self.window.assign(wnd)
    super(Schizo, self).build(input_shape)
  def call(self, x):
    if self.activation == 'relu':
      return(K.relu(K.dot(x, self.kernel * self.window) + self.bias))
    elif self.activation == 'softmax':
      return(K.softmax(K.dot(x, self.kernel * self.window) + self.bias))
    else:
      return(K.dot(x, self.kernel * self.window) + self.bias)
  def compute_output_shape(self, input_shape):
    return(input_shape[0], self.output_dim)
  def get_num_zeros(self):
    return(self.num_weights - self.num_ones)
  def get_num_weights(self):
    return(self.num_weights)
  def get_reduced_ratio(self):
    return(self.reduced_ratio)
  def get_halfwidth(self):
    return(self.halfwidth)
#end class Schizo

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
ilist = 0
for id in idlist:
  fcount=0
  if id > idim/2:
    continue
  irepeat = 0
  for irpt in range(num_repeat):
    # schizophrenia mimicking layer: (ilayer+1)-th layer
    ilayer = 9
    if output_mode != 0:
      print('id=', id, 'irepeat=', irepeat, '/', num_repeat)
    model = keras.Sequential([
      layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal', input_shape=training_images.shape[1:]),
      layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(0.25),
      layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(0.25),
      layers.Flatten(),
      #layers.Dense(idim, activation='relu', kernel_initializer='he_normal'),
      #layers.Dropout(0.5),
      Schizo(idim, reduction_ratio=id, form='diagonal', activation='relu', kernel_initializer='he_normal'),
      layers.Dense(num_class, activation='softmax')
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
    scores = model.evaluate(test_images, test_labels, verbose=output_mode)
    if ilayer >= 0:
      evalout[ilist][1] = model.layers[ilayer].get_reduced_ratio()
    evalout[ilist][2] += scores[0]
    evalout[ilist][3] += scores[1]
    irepeat += 1
  ###for irpt
  evalout[ilist][0] = id
  if irepeat > 0:
    evalout[ilist][2] /= irepeat
    evalout[ilist][3] /= irepeat
  ###if
  with open(path, mode='a') as flog:
    print('id=', evalout[ilist][0], 'frac=', evalout[ilist][1], 'loss=', evalout[ilist][2], 'acc=', evalout[ilist][3], file=flog)
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
  print('id\t','fraction\t','loss\t','accuracy', file=flog)
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2], '\t', evalout[ie][3], file=flog)
  #for ie
#open

if output_mode != 0:
  print(stitle)
  print('seed\t', 'dim\t', 'epoch\t', 'repeat')
  print(randomseed, '\t', idim, '\t', num_epoch, '\t', num_repeat)
  print('id\t','fraction\t','loss\t','accuracy')
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2], '\t', evalout[ie][3])
  #for ie
#endif


