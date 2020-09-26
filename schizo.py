from tensorflow.keras import layers
from tensorflow.keras import backend as K
import math
import numpy as np

#https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/layers/core.py
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
class SzDense(layers.Layer):
  def __init__(self,
               units,
               halfbandwidth=0, 
               param_reduction=0.5, 
               form='diagonal', 
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(SzDense, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True

    self.halfbandwidth = halfbandwidth
    self.form = form
    self.reduction_sv = param_reduction
    self.num_ones = 0
    self.reduced_ratio = 0
    self.num_weights = 0
    self.reduced_ratio = 0

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `SzDense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `SzDense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self.window = self.add_weight(name='window',
                                  shape=[last_dim, self.units],
                                  initializer='ones',
                                  trainable=False)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None

    #window init
    self.num_ones = 0
    self.reduced_ratio = 0
    nx = last_dim
    ny = self.units
    self.num_weights = nx * ny
    if self.halfbandwidth == 0:
      self.halfbandwidth = (nx*ny / math.sqrt(nx*nx + ny*ny)) * (1. - math.sqrt(self.reduction_sv)) 
      if self.form == 'gaussian':
        self.halfbandwidth *= 1.5
    #endif
    wnd = np.zeros((nx,ny))
    w_corr = 1.
    if self.form == 'diagonal':
      if ny > 1:
        rxy = (nx-1) / (ny-1)
        hwdiv = self.halfbandwidth * math.sqrt(rxy * rxy + 1)
        for iy in range(ny):
          ix1 = rxy * iy - hwdiv
          ix1 = int(ix1) + 1 if ix1 >= 0 else 0
          if ix1 > nx-1:
            continue
          ix2 = rxy * iy + hwdiv
          ix2 = math.ceil(ix2) if ix2 < nx else nx
          wnd[ix1:ix2, iy:iy+1] = 1
          self.num_ones += (ix2-ix1)
        #for ixiy
      else:
        wnd[:,:] = 1
        self.num_ones += nx
      #endif ny>1
      self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
      if self.num_ones > 0:
        w_corr = self.num_weights / self.num_ones
      self.kernel.assign(self.kernel * (wnd * w_corr))
    elif self.form == 'gaussian':
      if (self.halfbandwidth > 0) and (ny > 1):
        sgm2 = 1. / (2. * self.halfbandwidth * self.halfbandwidth)
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
      #endif halfbandwidth
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

    self.built = True

  def call(self, inputs):
    return core_ops.dense(
        inputs,
        self.kernel * self.window,
        self.bias,
        self.activation,
        dtype=self._compute_dtype_object)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(SzDense, self).get_config()
    config.update({
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    })
    return config
  def get_num_zeros(self):
    return(self.num_weights - self.num_ones)
  def get_num_weights(self):
    return(self.num_weights)
  def get_reduced_ratio(self):
    return(self.reduced_ratio)
  def get_halfbandwidth(self):
    return(self.halfbandwidth)
#class SzDense

from tensorflow.python.keras.utils.conv_utils import conv_output_length
import random
class SzConv2D(layers.Layer):
  def __init__(self, filters, kernel_size, param_reduction=0.5, form='individual', activation=None,
               padding='valid', strides=1, dilation_rate=1, kernel_initializer='glorot_uniform', 
               **kwargs):
    self.reduction_sv = param_reduction
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

