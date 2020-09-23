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

stitle = 'CIFAR10_VGG16Schizo200921act'
num_repeat = 3
num_epoch = 200
randomseed = 5
num_batch = 200
output_mode = 1
#output_mode = 0: no console output (minimum output)
#output_mode = 1: show progress bar (jupyter notebook)
#output_mode = 2: one line per epoch (shell script)

idlist = [0.6]
len_idlist = len(idlist)
evalout = [[0] * 4 for i in range(len_idlist)]

class SzDense(layers.Layer):
  def __init__(self, output_dim, halfwidth=0, reduction_ratio=0.5, form='diagonal', activation='relu', kernel_initializer='he_normal', **kwargs):
    self.output_dim = output_dim
    self.halfwidth = halfwidth
    self.form = form
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.reduction_sv = reduction_ratio
    super(SzDense, self).__init__(**kwargs)
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
    elif self.form == 'random':
      wnd = np.random.rand(nx,ny)
      wnd = np.where(wnd < self.reduction_sv, 0, 1)
      self.num_ones = np.sum(wnd)
      self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
      if self.num_ones > 0:
        w_corr = self.num_weights / self.num_ones
      self.kernel.assign(self.kernel * (wnd * w_corr))
    #endif form_function
    self.window.assign(wnd)
    super(SzDense, self).build(input_shape)
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
#end class SzDense

from tensorflow.python.keras.utils.conv_utils import conv_output_length
import random
class SzConv2D(layers.Layer):
  def __init__(self, filters, kernel_size, reduction_ratio=0.5, form='individual', activation=None,
               padding='valid', strides=1, dilation_rate=1, kernel_initializer='glorot_uniform', 
               **kwargs):
    self.reduction_sv = reduction_ratio
    self.kernel_initializer = kernel_initializer
    self.strides = strides
    self.padding = padding
    self.dilation_rate = dilation_rate
    self.filters = filters
    self.activation = activation
    self.form = form
    self.kernel_size = kernel_size
    super(SzConv2D, self).__init__(**kwargs)
  def build(self, input_shape):
    self.num_ones = 0
    ksz0 = self.kernel_size[0]
    ksz1 = self.kernel_size[1]
    nx = input_shape[-1]
    ny = self.filters
    ksz = ksz0 * ksz1
    self.num_weights = ksz * nx * ny
    self.kernel = self.add_weight(name='kernel',
                                  shape=(ksz0, ksz1, nx, ny),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
    self.window = self.add_weight(name='window', 
                                  shape=(ksz0, ksz1, nx, ny), 
                                  initializer='ones', 
                                  trainable=False)
    self.bias   = self.add_weight(name='bias',
                                  shape=(1, ny),
                                  initializer='zeros',
                                  trainable=True)
    wnd = np.zeros((ksz0, ksz1, nx, ny))
    w_corr = 1.
    if self.form == 'individual':
      wnd = np.random.rand(ksz0, ksz1, nx, ny)
      wnd = np.where(wnd < self.reduction_sv, 0, 1)
      self.num_ones = np.sum(wnd)
    elif self.form == 'kernel':
      for ix in range(nx):
        for iy in range(ny):
          if random.random() > self.reduction_sv:
            wnd[:, :, ix, iy] = 1
            self.num_ones += ksz
    #endif self.form
    self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
    if self.num_ones > 0:
      w_corr = self.num_weights / self.num_ones
    self.kernel.assign(self.kernel * (wnd * w_corr))
    self.window.assign(wnd)
    super(SzConv2D, self).build(input_shape)
  def call(self, x):
    if self.activation == 'relu':
      return K.relu(K.conv2d(x, 
                             (self.kernel * self.window), 
                             strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate)
                    + self.bias)
    elif self.activation == 'softmax':
      return K.softmax(K.conv2d(x, 
                                (self.kernel * self.window),  
                                strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate)
                       + self.bias)
    else:
      return (K.conv2d(x, 
                       (self.kernel * self.window),  
                       strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate) 
              + self.bias)
    #return super(SzConv2D, self).call(x)
  def compute_output_shape(self, input_shape):
    length = conv_output_length(input_shape[1], self.kernel_size[0], self.padding, self.strides[0], dilation=self.dilation_rate[0])
    return (input_shape[0], length, self.filters)
  def get_num_zeros(self):
    return(self.num_weights - self.num_ones)
  def get_num_weights(self):
    return(self.num_weights)
  def get_reduced_ratio(self):
    return(self.reduced_ratio)
#SzConv2D

def Learning_rate_by_step(epoch):
  lrnRate = 0.0005
  if(epoch >= 150):
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
    ilayer = 28
    # ilayer = 34
    if output_mode != 0:
      print('id=', id, 'irepeat=', irepeat, '/', num_repeat)
    model = keras.Sequential([
      layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=training_images.shape[1:], kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'), 
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      #
      layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      #
      layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(),
      #
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(),
      #
      SzConv2D(filters=512, reduction_ratio=0.0, kernel_size=(3,3), padding='same', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      SzConv2D(filters=512, reduction_ratio=0.0, kernel_size=(3,3), padding='same', kernel_initializer='he_normal'),
      layers.BatchNormalization(),
      layers.Activation('relu'),
      SzConv2D(filters=512, reduction_ratio=id, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      #
      layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
      #layers.Dropout(0.5),
      layers.Dense(4096, activation='relu', kernel_initializer='he_normal'),
      #layers.Dropout(0.5),
      layers.Dense(1024, activation='relu', kernel_initializer='he_normal'),
      #layers.Dropout(0.5),
      #SzDense(4096, reduction_ratio=0.0, activation='relu', kernel_initializer='he_normal'),
      #SzDense(4096, reduction_ratio=0.0, activation='relu', kernel_initializer='he_normal'),
      #SzDense(1024, reduction_ratio=0.0, activation='relu', kernel_initializer='he_normal'),
      layers.Dense(num_class, activation='softmax')
    ])
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
      evalout[ilist][0] = model.layers[ilayer].get_num_zeros()
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
  print(randomseed, '\t', 0, '\t', num_epoch, '\t', num_repeat, file=flog)
  print('width\t','reduction\t','loss\t','accuracy', file=flog)
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2], '\t', evalout[ie][3], file=flog)
  #for ie
#open

if output_mode != 0:
  print(stitle)
  print('seed\t', 'dim\t', 'epoch\t', 'repeat')
  print(randomseed, '\t', 0, '\t', num_epoch, '\t', num_repeat)
  print('width\t','reduction\t','loss\t','accuracy')
  for ie in range(len_idlist):
    print(evalout[ie][0], '\t', evalout[ie][1], '\t', evalout[ie][2], '\t', evalout[ie][3])
  #for ie
#endif



