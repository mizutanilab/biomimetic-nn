from tensorflow.keras import layers
from tensorflow.keras import backend as K
import math
import numpy as np

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
