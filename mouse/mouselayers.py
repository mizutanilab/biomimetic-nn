#https://github.com/keras-team/keras/blob/master/keras/src/layers/core/dense.py

import ml_dtypes

from keras.src import activations
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer

import math
import numpy as np

@keras_export("keras.layers.Dense")
class mDense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors). The output in this case will have
    shape `(batch_size, d0, units)`.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `Dense` layer by calling `layer.enable_lora(rank)`.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        halfbandwidth=0, 
        param_reduction=0.5, 
        form='diagonal', 
        input2d_width='10', 
        output2d_width='10',
        window2d_width='1.41', 

        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_enabled = False
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        self.halfbandwidth = halfbandwidth
        self.form = form
        self.reduction_sv = param_reduction
        self.input2d_width = input2d_width
        self.output2d_width = output2d_width
        self.window2d_width = window2d_width
        self.num_ones = 0
        self.reduced_ratio = 0
        self.num_weights = 0
        self.reduced_ratio = 0

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # We use `self._dtype_policy` to check to avoid issues in torch dynamo
        is_quantized = isinstance(
            self._dtype_policy, dtype_policies.QuantizedDTypePolicy
        )
        if is_quantized:
            self.quantized_build(
                input_shape, mode=self.dtype_policy.quantization_mode
            )
        if not is_quantized or self.dtype_policy.quantization_mode != "int8":
            # If the layer is quantized to int8, `self._kernel` will be added
            # in `self._int8_build`. Therefore, we skip it here.
            self._kernel = self.add_weight(
                name="kernel",
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        #window init
        self.num_ones = 0
        self.reduced_ratio = 0
        nx = input_dim
        ny = self.units
        self.num_weights = nx * ny
        if self.halfbandwidth == 0:
          self.halfbandwidth = (nx*ny / math.sqrt(nx*nx + ny*ny)) * (1. - math.sqrt(self.reduction_sv)) 
          if self.form == 'gaussian':
            self.halfbandwidth *= 1.5
        #endif
        self.wnd = np.zeros((nx,ny))
        self.w_corr = 1.
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
              self.wnd[ix1:ix2, iy:iy+1] = 1
              self.num_ones += (ix2-ix1)
            #for ixiy
          else:
            self.wnd[:,:] = 1
            self.num_ones += nx
          #endif ny>1
          self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
          if self.num_ones > 0:
            self.w_corr = self.num_weights / self.num_ones
          self._kernel.assign(self._kernel * (self.wnd * self.w_corr))
        elif self.form == 'gaussian':
          if (self.halfbandwidth > 0) and (ny > 1):
            sgm2 = 1. / (2. * self.halfbandwidth * self.halfbandwidth)
            gsum = 0
            rxy = (nx-1) / (ny-1)
            for ix in range(nx):
              for iy in range(ny):
                gauss = math.exp(-(ix-rxy*iy)*(ix-rxy*iy)*sgm2)
                self.wnd[ix][iy] = gauss
                gsum += gauss
            #for ixiy
            self.reduced_ratio = 1. - gsum / self.num_weights
            if gsum > 0:
              self.w_corr = self.num_weights / gsum
            self.wnd = self.wnd * self.w_corr
          else:
            self.wnd[:,:] = 1
            self.num_ones = nx * ny
          #endif halfbandwidth
          self._kernel.assign(self._kernel * self.wnd)
        elif self.form == '2d':
          if ny > 1:
            nx1 = self.input2d_width
            nx2 = nx // self.input2d_width
            ny1 = self.output2d_width
            ny2 = ny // self.output2d_width
            d1 = self.window2d_width
            d2 = self.window2d_width * self.window2d_width
            #print('2d', nx1, nx2, ny1, ny2, d1, d2)
            #####original precise but slow version 240401
            #for ix in range(nx):
            #  for iy in range(ny):
            #    dx = (ix % nx1) / nx1 - (iy % ny1) / ny1
            #    dy = (ix // nx1) / nx2 - (iy // ny1) / ny2
            #    if (dx * dx + dy * dy < d2): 
            #      wnd[ix][iy] = 1
            #      self.num_ones += 1
            #    #endif
            #####integer version 240406
            for ix in range(nx):
              ox = (ix % nx1) / nx1
              oy = (ix // nx1) / nx2
              oymin = max(math.ceil((oy - d1) * ny2), 0)
              oymax = min(math.ceil((oy + d1) * ny2), ny2)
              for ky in range(oymin, oymax):
                dx = d2 - (ky/ny2 - oy) * (ky/ny2 - oy)
                if dx > 0:
                  dx = math.sqrt(dx)
                  oxmin = max(math.ceil((ox - dx) * ny1), 0)
                  oxmax = min(math.ceil((ox + dx) * ny1), ny1)
                  if (oxmax > oxmin) :
                    self.wnd[ix, (ky*ny1+oxmin):(ky*ny1+oxmax)] = 1
                    self.num_ones += oxmax - oxmin
                  #endif
                #endif dx > 0
              #for ky
            #for ixiy
          else:
            self.wnd[:,:] = 1
            self.num_ones += nx
          #endif ny>1
          self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
          if self.num_ones > 0:
            self.w_corr = self.num_weights / self.num_ones
          self._kernel.assign(self._kernel * (self.wnd * self.w_corr))
          #print('2d', self.num_weights, self.num_ones)
        elif self.form == 'random':
          self.wnd = np.random.rand(nx,ny)
          self.wnd = np.where(self.wnd < self.reduction_sv, 0, 1)
          self.num_ones = np.sum(self.wnd)
          self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
          if self.num_ones > 0:
            self.w_corr = self.num_weights / self.num_ones
          self._kernel.assign(self._kernel * (self.wnd * self.w_corr))
        #endif form_function
        #240509 kernel.assign does not work here
        #self.window.assign(self.wnd)

        #test codes
        #self.window.assign(ops.ones(shape=(3072, 1000)))
        #print(self.window)
        #print(ops.ones(shape=(3072, 1000)))
        
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        if self.lora_enabled:
            return self._kernel + ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )
        return self._kernel

    def call(self, inputs, training=None):
        #x = ops.matmul(inputs, self.kernel)
        x = ops.matmul(inputs, self.kernel * self.wnd)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    #def enable_lora(
    #    self, rank, a_initializer="he_uniform", b_initializer="zeros"
    #):
    #    if self.kernel_constraint:
    #        raise ValueError(
    #            "Lora is incompatible with kernel constraints. "
    #            "In order to enable lora on this layer, remove the "
    #            "`kernel_constraint` argument."
    #        )
    #    if not self.built:
    #        raise ValueError(
    #            "Cannot enable lora on a layer that isn't yet built."
    #        )
    #    if self.lora_enabled:
    #        raise ValueError(
    #            "lora is already enabled. "
    #            "This can only be done once per layer."
    #        )
    #    self._tracker.unlock()
    #    self.lora_kernel_a = self.add_weight(
    #        name="lora_kernel_a",
    #        shape=(self.kernel.shape[0], rank),
    #        initializer=initializers.get(a_initializer),
    #        regularizer=self.kernel_regularizer,
    #    )
    #    self.lora_kernel_b = self.add_weight(
    #        name="lora_kernel_b",
    #        shape=(rank, self.kernel.shape[1]),
    #        initializer=initializers.get(b_initializer),
    #        regularizer=self.kernel_regularizer,
    #    )
    #    self._kernel.trainable = False
    #    self._tracker.lock()
    #    self.lora_enabled = True
    #    self.lora_rank = rank

    #def save_own_variables(self, store):
    #    # Do nothing if the layer isn't yet built
    #    if not self.built:
    #        return
    #    # The keys of the `store` will be saved as determined because the
    #    # default ordering will change after quantization
    #    kernel_value, kernel_scale = self._get_kernel_with_merged_lora()
    #    target_variables = [kernel_value]
    #    if self.use_bias:
    #        target_variables.append(self.bias)
    #    if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
    #        mode = self.dtype_policy.quantization_mode
    #        if mode == "int8":
    #            target_variables.append(kernel_scale)
    #        elif mode == "float8":
    #            target_variables.append(self.inputs_scale)
    #            target_variables.append(self.inputs_amax_history)
    #            target_variables.append(self.kernel_scale)
    #            target_variables.append(self.kernel_amax_history)
    #            target_variables.append(self.outputs_grad_scale)
    #            target_variables.append(self.outputs_grad_amax_history)
    #        else:
    #            raise NotImplementedError(
    #                self.QUANTIZATION_MODE_ERROR_TEMPLATE.format(mode=mode)
    #            )
    #    for i, variable in enumerate(target_variables):
    #        store[str(i)] = variable

    #def load_own_variables(self, store):
    #    if not self.lora_enabled:
    #        self._check_load_own_variables(store)
    #    # Do nothing if the layer isn't yet built
    #    if not self.built:
    #        return
    #    # The keys of the `store` will be saved as determined because the
    #    # default ordering will change after quantization
    #    target_variables = [self._kernel]
    #    if self.use_bias:
    #        target_variables.append(self.bias)
    #    if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
    #        mode = self.dtype_policy.quantization_mode
    #        if mode == "int8":
    #            target_variables.append(self.kernel_scale)
    #        elif mode == "float8":
    #            target_variables.append(self.inputs_scale)
    #            target_variables.append(self.inputs_amax_history)
    #            target_variables.append(self.kernel_scale)
    #            target_variables.append(self.kernel_amax_history)
    #            target_variables.append(self.outputs_grad_scale)
    #            target_variables.append(self.outputs_grad_amax_history)
    #        else:
    #            raise NotImplementedError(
    #                self.QUANTIZATION_MODE_ERROR_TEMPLATE.format(mode=mode)
    #            )
    #    for i, variable in enumerate(target_variables):
    #        variable.assign(store[str(i)])
    #    if self.lora_enabled:
    #        self.lora_kernel_a.assign(ops.zeros(self.lora_kernel_a.shape))
    #        self.lora_kernel_b.assign(ops.zeros(self.lora_kernel_b.shape))

    #def get_config(self):
    #    base_config = super().get_config()
    #    config = {
    #        "units": self.units,
    #        "activation": activations.serialize(self.activation),
    #        "use_bias": self.use_bias,
    #        "kernel_initializer": initializers.serialize(
    #            self.kernel_initializer
    #        ),
    #        "bias_initializer": initializers.serialize(self.bias_initializer),
    #        "kernel_regularizer": regularizers.serialize(
    #            self.kernel_regularizer
    #        ),
    #        "bias_regularizer": regularizers.serialize(self.bias_regularizer),
    #        "kernel_constraint": constraints.serialize(self.kernel_constraint),
    #        "bias_constraint": constraints.serialize(self.bias_constraint),
    #    }
    #    if self.lora_rank:
    #        config["lora_rank"] = self.lora_rank
    #    return {**base_config, **config}

    #def _check_load_own_variables(self, store):
    #    all_vars = self._trainable_variables + self._non_trainable_variables
    #    if len(store.keys()) != len(all_vars):
    #        if len(all_vars) == 0 and not self.built:
    #            raise ValueError(
    #                f"Layer '{self.name}' was never built "
    #                "and thus it doesn't have any variables. "
    #                f"However the weights file lists {len(store.keys())} "
    #                "variables for this layer.\n"
    #                "In most cases, this error indicates that either:\n\n"
    #                "1. The layer is owned by a parent layer that "
    #                "implements a `build()` method, but calling the "
    #                "parent's `build()` method did NOT create the state of "
    #                f"the child layer '{self.name}'. A `build()` method "
    #                "must create ALL state for the layer, including "
    #                "the state of any children layers.\n\n"
    #                "2. You need to implement "
    #                "the `def build_from_config(self, config)` method "
    #                f"on layer '{self.name}', to specify how to rebuild "
    #                "it during loading. "
    #                "In this case, you might also want to implement the "
    #                "method that generates the build config at saving time, "
    #                "`def get_build_config(self)`. "
    #                "The method `build_from_config()` is meant "
    #                "to create the state "
    #                "of the layer (i.e. its variables) upon deserialization.",
    #            )
    #        raise ValueError(
    #            f"Layer '{self.name}' expected {len(all_vars)} variables, "
    #            "but received "
    #            f"{len(store.keys())} variables during loading. "
    #            f"Expected: {[v.name for v in all_vars]}"
    #        )

    # Quantization-related (int8 and float8) methods

    #QUANTIZATION_MODE_ERROR_TEMPLATE = (
    #    f"Invalid quantization mode. Expected one of "
    #    f"{dtype_policies.QUANTIZATION_MODES}. "
    #    "Received: quantization_mode={mode}"
    #)

    #def quantized_build(self, input_shape, mode):
    #    if mode == "int8":
    #        input_dim = input_shape[-1]
    #        kernel_shape = (input_dim, self.units)
    #        self._int8_build(kernel_shape)
    #    elif mode == "float8":
    #        self._float8_build()
    #    else:
    #        raise NotImplementedError(
    #            self.QUANTIZATION_MODE_ERROR_TEMPLATE.format(mode=mode)
    #        )

    #def _int8_build(
    #    self,
    #    kernel_shape,
    #    kernel_initializer="zeros",
    #    kernel_scale_initializer="ones",
    #):
    #    self.inputs_quantizer = quantizers.AbsMaxQuantizer(axis=-1)
    #    self._kernel = self.add_weight(
    #        name="kernel",
    #        shape=kernel_shape,
    #        initializer=kernel_initializer,
    #        dtype="int8",
    #        trainable=False,
    #    )
    #    self.kernel_scale = self.add_weight(
    #        name="kernel_scale",
    #        shape=(self.units,),
    #        initializer=kernel_scale_initializer,
    #        trainable=False,
    #    )
    #    self._is_quantized = True

    #def _float8_build(self):
    #    from keras.src.dtype_policies import QuantizedFloat8DTypePolicy

    #    # If `self.dtype_policy` is not QuantizedFloat8DTypePolicy, then set
    #    # `amax_history_length` to its default value.
    #    amax_history_length = getattr(
    #        self.dtype_policy,
    #        "amax_history_length",
    #        QuantizedFloat8DTypePolicy.default_amax_history_length,
    #    )
    #    # We set `trainable=True` because we will use the gradients to overwrite
    #    # these variables
    #    scale_kwargs = {
    #        "shape": (),
    #        "initializer": "ones",
    #        "dtype": "float32",  # Always be float32
    #        "trainable": True,
    #        "autocast": False,
    #    }
    #    amax_history_kwargs = {
    #        "shape": (amax_history_length,),
    #        "initializer": "zeros",
    #        "dtype": "float32",  # Always be float32
    #        "trainable": True,
    #        "autocast": False,
    #    }
    #    self.inputs_scale = self.add_weight(name="inputs_scale", **scale_kwargs)
    #    self.inputs_amax_history = self.add_weight(
    #        name="inputs_amax_history", **amax_history_kwargs
    #    )
    #    self.kernel_scale = self.add_weight(name="kernel_scale", **scale_kwargs)
    #    self.kernel_amax_history = self.add_weight(
    #        name="kernel_amax_history", **amax_history_kwargs
    #    )
    #    self.outputs_grad_scale = self.add_weight(
    #        name="outputs_grad_scale", **scale_kwargs
    #    )
    #    self.outputs_grad_amax_history = self.add_weight(
    #        name="outputs_grad_amax_history", **amax_history_kwargs
    #    )
    #    # We need to set `overwrite_with_gradient=True` to instruct the
    #    # optimizer to directly overwrite these variables with their computed
    #    # gradients during training
    #    self.inputs_scale.overwrite_with_gradient = True
    #    self.inputs_amax_history.overwrite_with_gradient = True
    #    self.kernel_scale.overwrite_with_gradient = True
    #    self.kernel_amax_history.overwrite_with_gradient = True
    #    self.outputs_grad_scale.overwrite_with_gradient = True
    #    self.outputs_grad_amax_history.overwrite_with_gradient = True
    #    self._is_quantized = True

    def quantized_call(self, inputs, training=None):

        if self.dtype_policy.quantization_mode == "int8":
            return self._int8_call(inputs)
        elif self.dtype_policy.quantization_mode == "float8":
            return self._float8_call(inputs, training=training)
        else:
            mode = self.dtype_policy.quantization_mode
            raise NotImplementedError(
                self.QUANTIZATION_MODE_ERROR_TEMPLATE.format(mode=mode)
            )

    def _int8_call(self, inputs):
        @ops.custom_gradient
        def matmul_with_inputs_gradient(inputs, kernel, kernel_scale):
            def grad_fn(*args, upstream=None):
                if upstream is None:
                    (upstream,) = args
                float_kernel = ops.divide(
                    ops.cast(kernel, dtype=self.compute_dtype),
                    kernel_scale,
                )
                inputs_grad = ops.matmul(upstream, ops.transpose(float_kernel))
                return (inputs_grad, None, None)

            inputs, inputs_scale = self.inputs_quantizer(inputs)
            x = ops.matmul(inputs, kernel)
            # De-scale outputs
            x = ops.cast(x, self.compute_dtype)
            x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
            return x, grad_fn

        x = matmul_with_inputs_gradient(
            inputs,
            #ops.convert_to_tensor(self._kernel),
            ops.convert_to_tensor(self._kernel * self.wnd),
            ops.convert_to_tensor(self.kernel_scale),
        )
        if self.lora_enabled:
            lora_x = ops.matmul(inputs, self.lora_kernel_a)
            lora_x = ops.matmul(lora_x, self.lora_kernel_b)
            x = ops.add(x, lora_x)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _float8_call(self, inputs, training=None):
        if self.lora_enabled:
            raise NotImplementedError(
                "Currently, `_float8_call` doesn't support LoRA"
            )

        @ops.custom_gradient
        def quantized_dequantize_inputs(inputs, scale, amax_history):
            if training:
                new_scale = quantizers.compute_float8_scale(
                    ops.max(amax_history, axis=0),
                    scale,
                    ops.cast(
                        float(ml_dtypes.finfo("float8_e4m3fn").max), "float32"
                    ),
                )
                new_amax_history = quantizers.compute_float8_amax_history(
                    inputs, amax_history
                )
            else:
                new_scale = None
                new_amax_history = None
            qdq_inputs = quantizers.quantize_and_dequantize(
                inputs, scale, "float8_e4m3fn", self.compute_dtype
            )

            def grad(*args, upstream=None, variables=None):
                if upstream is None:
                    (upstream,) = args
                return upstream, new_scale, new_amax_history

            return qdq_inputs, grad

        @ops.custom_gradient
        def quantized_dequantize_outputs(outputs, scale, amax_history):
            """Quantize-dequantize the output gradient but not the output."""

            def grad(*args, upstream=None, variables=None):
                if upstream is None:
                    (upstream,) = args
                new_scale = quantizers.compute_float8_scale(
                    ops.max(amax_history, axis=0),
                    scale,
                    ops.cast(
                        float(ml_dtypes.finfo("float8_e5m2").max), "float32"
                    ),
                )
                qdq_upstream = quantizers.quantize_and_dequantize(
                    upstream, scale, "float8_e5m2", self.compute_dtype
                )
                new_amax_history = quantizers.compute_float8_amax_history(
                    upstream, amax_history
                )
                return qdq_upstream, new_scale, new_amax_history

            return outputs, grad

        x = ops.matmul(
            quantized_dequantize_inputs(
                inputs,
                ops.convert_to_tensor(self.inputs_scale),
                ops.convert_to_tensor(self.inputs_amax_history),
            ),
            quantized_dequantize_inputs(
                #ops.convert_to_tensor(self._kernel),
                ops.convert_to_tensor(self._kernel * self.wnd),
                ops.convert_to_tensor(self.kernel_scale),
                ops.convert_to_tensor(self.kernel_amax_history),
            ),
        )
        # `quantized_dequantize_outputs` is placed immediately after
        # `ops.matmul` for the sake of pattern matching in gemm_rewrite. That
        # way, the qdq will be adjacent to the corresponding matmul_bprop in the
        # bprop.
        x = quantized_dequantize_outputs(
            x,
            ops.convert_to_tensor(self.outputs_grad_scale),
            ops.convert_to_tensor(self.outputs_grad_amax_history),
        )
        if self.bias is not None:
            # Under non-mixed precision cases, F32 bias has to be converted to
            # BF16 first to get the biasAdd fusion support. ref. PR
            # https://github.com/tensorflow/tensorflow/pull/60306
            bias = self.bias
            if self.dtype_policy.compute_dtype == "float32":
                bias_bf16 = ops.cast(bias, "bfloat16")
                bias = ops.cast(bias_bf16, bias.dtype)
            x = ops.add(x, bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    #def quantize(self, mode):
    #    import gc

    #    # Prevent quantization of the subclasses
    #    if type(self) is not mDense:
    #        raise NotImplementedError(
    #            f"Layer {self.__class__.__name__} does not have a `quantize()` "
    #            "method implemented."
    #        )
    #    self._check_quantize_args(mode, self.compute_dtype)

    #    self._tracker.unlock()
    #    if mode == "int8":
    #        # Quantize `self._kernel` to int8 and compute corresponding scale
    #        kernel_value, kernel_scale = quantizers.abs_max_quantize(
    #            self._kernel, axis=0
    #        )
    #        kernel_scale = ops.squeeze(kernel_scale, axis=0)
    #        self._untrack_variable(self._kernel)
    #        kernel_shape = self._kernel.shape
    #        del self._kernel
    #        # Utilize a lambda expression as an initializer to prevent adding a
    #        # large constant to the computation graph.
    #        self._int8_build(
    #            kernel_shape,
    #            lambda shape, dtype: kernel_value,
    #            lambda shape, dtype: kernel_scale,
    #        )
    #    elif mode == "float8":
    #        self._float8_build()
    #    else:
    #        raise NotImplementedError(
    #            self.QUANTIZATION_MODE_ERROR_TEMPLATE.format(mode=mode)
    #        )
    #    self._tracker.lock()

    #    # Set new dtype policy
    #    if not isinstance(
    #        self.dtype_policy, dtype_policies.QuantizedDTypePolicy
    #    ):
    #        quantized_dtype = f"{mode}_from_{self.dtype_policy.name}"
    #        # We set the internal `self._dtype_policy` instead of using the
    #        # setter to avoid double `quantize` call
    #        self._dtype_policy = dtype_policies.get(quantized_dtype)

    #    # Release memory manually because sometimes the backend doesn't
    #    gc.collect()

    #def _get_kernel_with_merged_lora(self):
    #    if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
    #        kernel_value = self._kernel
    #        kernel_scale = self.kernel_scale
    #        if self.lora_enabled:
    #            # Dequantize & quantize to merge lora weights into int8 kernel
    #            # Note that this is a lossy compression
    #            kernel_value = ops.divide(kernel_value, kernel_scale)
    #            kernel_value = ops.add(
    #                kernel_value,
    #                ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
    #            )
    #            kernel_value, kernel_scale = quantizers.abs_max_quantize(
    #                kernel_value, axis=0
    #            )
    #            kernel_scale = ops.squeeze(kernel_scale, axis=0)
    #        return kernel_value, kernel_scale
    #    return self.kernel, None

    def get_num_zeros(self):
        return(self.num_weights - self.num_ones)
    def get_num_weights(self):
        return(self.num_weights)
    def get_reduced_ratio(self):
        return(self.reduced_ratio)
    def get_halfbandwidth(self):
        return(self.halfbandwidth)
#class mDense

#https://github.com/keras-team/keras/blob/master/keras/src/layers/convolutional/base_conv.py
"""Keras base class for convolution layers."""

from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.backend import standardize_data_format
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.ops.operation_utils import compute_conv_output_shape
from keras.src.utils.argument_validation import standardize_padding
from keras.src.utils.argument_validation import standardize_tuple

class mBaseConv(Layer):
    """Abstract N-D convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved (actually
    cross-correlated) with the layer input to produce a tensor of outputs. If
    `use_bias` is True (and a `bias_initializer` is provided), a bias vector is
    created and added to the outputs. Finally, if `activation` is not `None`, it
    is applied to the outputs as well.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Args:
        rank: int, the rank of the convolution, e.g. 2 for 2D convolution.
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of `rank` integers, specifying the size
            of the convolution window.
        strides: int or tuple/list of `rank` integers, specifying the stride
            length of the convolution. If only one int is specified, the same
            stride size will be used for all dimensions. `strides > 1` is
            incompatible with `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of `rank` integers, specifying the
            dilation rate to use for dilated convolution. If only one int is
            specified, the same dilation rate will be used for all dimensions.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters // groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing layer by calling
            `layer.enable_lora(rank)`.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,

        halfbandwidth=0, 
        param_reduction=0.5, 
        form='diagonal', 
        input2d_width='10', 
        output2d_width='10',
        window2d_width='1.41', 

        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.rank = rank
        self.filters = filters
        self.groups = groups
        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.dilation_rate = standardize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.padding = standardize_padding(padding, allow_causal=rank == 1)
        self.data_format = standardize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_enabled = False
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.data_format = self.data_format

        self.halfbandwidth = halfbandwidth
        self.form = form
        self.reduction_sv = param_reduction
        self.input2d_width = input2d_width
        self.output2d_width = output2d_width
        self.window2d_width = window2d_width
        self.num_ones = 0
        self.reduced_ratio = 0
        self.num_weights = 0
        self.reduced_ratio = 0

        if self.filters is not None and self.filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. Expected a strictly "
                f"positive value. Received filters={self.filters}."
            )

        if self.groups <= 0:
            raise ValueError(
                "The number of groups must be a positive integer. "
                f"Received: groups={self.groups}."
            )

        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the "
                f"number of groups. Received: groups={self.groups}, "
                f"filters={self.filters}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0. Received "
                f"kernel_size={self.kernel_size}."
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0. Received "
                f"strides={self.strides}"
            )

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                "`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            channel_axis = 1
            input_channel = input_shape[1]
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        if input_channel % self.groups != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by "
                f"the number of groups. Received groups={self.groups}, but the "
                f"input has {input_channel} channels (full input shape is "
                f"{input_shape})."
            )
        kernel_shape = self.kernel_size + (
            input_channel // self.groups,
            self.filters,
        )

        # compute_output_shape contains some validation logic for the input
        # shape, and make sure the output shape has all positive dimensions.
        self.compute_output_shape(input_shape)

        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        #window initialization
        self.wnd = np.zeros(kernel_shape)
        self.w_corr = 1.
        nx = input_channel // self.groups
        #ny = input_channel
        ny = self.filters
        #nx = self.filters
        #print(self.wnd)
        #print('nxny', nx, ny)
        #print(kernel_shape)
        if self.form == 'individual':
          self.wnd = np.random.random_sample(kernel_shape)
          self.wnd = np.where(self.wnd < self.reduction_sv, 0, 1)
        elif self.form == 'kernel':
          for ix in range(nx):
            for iy in range(ny):
              if random.random() > self.reduction_sv:
                self.wnd[..., ix, iy] = 1
        elif self.form == 'diagonal':
          self.halfbandwidth = (nx*ny / math.sqrt(nx*nx + ny*ny)) * (1. - math.sqrt(self.reduction_sv)) 
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
              self.wnd[..., ix1:ix2, iy:iy+1] = 1
            #for ixiy
          else:
            self.wnd = np.ones(kernel_shape)
          #endif ny>1
        elif self.form == '2d':
          if ny > 1:
            nx1 = self.input2d_width
            nx2 = nx // self.input2d_width
            ny1 = self.output2d_width
            ny2 = ny // self.output2d_width
            d1 = self.window2d_width
            d2 = self.window2d_width * self.window2d_width
            #print('2d', nx1, nx2, ny1, ny2, d1, d2)
            #####original precise but slow version 240401
            #for ix in range(nx):
            #  for iy in range(ny):
            #    dx = (ix % nx1) / nx1 - (iy % ny1) / ny1
            #    dy = (ix // nx1) / nx2 - (iy // ny1) / ny2
            #    if (dx * dx + dy * dy < d2): 
            #      wnd[ix][iy] = 1
            #      self.num_ones += 1
            #    #endif
            #####integer version 240406
            for ix in range(nx):
              ox = (ix % nx1) / nx1
              oy = (ix // nx1) / nx2
              oymin = max(math.ceil((oy - d1) * ny2), 0)
              oymax = min(math.ceil((oy + d1) * ny2), ny2)
              for ky in range(oymin, oymax):
                dx = d2 - (ky/ny2 - oy) * (ky/ny2 - oy)
                if dx > 0:
                  dx = math.sqrt(dx)
                  oxmin = max(math.ceil((ox - dx) * ny1), 0)
                  oxmax = min(math.ceil((ox + dx) * ny1), ny1)
                  if (oxmax > oxmin) :
                    self.wnd[..., ix, (ky*ny1+oxmin):(ky*ny1+oxmax)] = 1
                    #self.num_ones += oxmax - oxmin
                  #endif
                #endif dx > 0
              #for ky
            #for ixiy
          else:
            #self.wnd[:,:] = 1
            self.wnd = np.ones(kernel_shape)
            #self.num_ones += nx
          #endif ny>1
        #endif self.form
        self.num_ones = np.sum(self.wnd)
        self.num_weights = self.wnd.size
        self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
        if self.num_ones > 0:
          self.w_corr = self.num_weights / self.num_ones
        self._kernel.assign(self._kernel * (self.wnd * self.w_corr))
        #self.window.assign(self.wnd)

        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        if self.lora_enabled:
            return self._kernel + ops.matmul(
                self.lora_kernel_a, self.lora_kernel_b
            )
        return self._kernel

    def convolution_op(self, inputs, kernel):
        return ops.conv(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

    def call(self, inputs):
        outputs = self.convolution_op(
            inputs,
            self.kernel * self.wnd,
        )
        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def enable_lora(
        self, rank, a_initializer="he_uniform", b_initializer="zeros"
    ):
        if self.kernel_constraint:
            raise ValueError(
                "Lora is incompatible with kernel constraints. "
                "In order to enable lora on this layer, remove the "
                "`kernel_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. "
                "This can only be done once per layer."
            )
        self._tracker.unlock()
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=self._kernel.shape[:-1] + (rank,),
            initializer=initializers.get(a_initializer),
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.filters),
            initializer=initializers.get(b_initializer),
            regularizer=self.kernel_regularizer,
        )
        self._kernel.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank

    def save_own_variables(self, store):
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        target_variables = [self.kernel]
        if self.use_bias:
            target_variables.append(self.bias)
        for i, variable in enumerate(target_variables):
            store[str(i)] = variable

    def load_own_variables(self, store):
        if not self.lora_enabled:
            self._check_load_own_variables(store)
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        target_variables = [self._kernel]
        if self.use_bias:
            target_variables.append(self.bias)
        for i, variable in enumerate(target_variables):
            variable.assign(store[str(i)])
        if self.lora_enabled:
            self.lora_kernel_a.assign(ops.zeros(self.lora_kernel_a.shape))
            self.lora_kernel_b.assign(ops.zeros(self.lora_kernel_b.shape))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
        return config

    def _check_load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )
    def get_num_zeros(self):
        return(self.num_weights - self.num_ones)
    def get_num_weights(self):
        return(self.num_weights)
    def get_reduced_ratio(self):
        return(self.reduced_ratio)
    def get_halfbandwidth(self):
        return(self.halfbandwidth)
#-----

#https://github.com/keras-team/keras/blob/master/keras/src/layers/convolutional/conv2d.py
from keras.src.api_export import keras_export
#from keras.src.layers.convolutional.base_conv import BaseConv


@keras_export(["keras.layers.mConv2D", "keras.layers.mConvolution2D"])
class mConv2D(mBaseConv):
    """2D convolution layer.

    This layer creates a convolution kernel that is convolved with the layer
    input over a 2D spatial (or temporal) dimension (height and width) to
    produce a tensor of outputs. If `use_bias` is True, a bias vector is created
    and added to the outputs. Finally, if `activation` is not `None`, it is
    applied to the outputs as well.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of 2 integer, specifying the size of the
            convolution window.
        strides: int or tuple/list of 2 integer, specifying the stride length
            of the convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dilation_rate: int or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters // groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`

    Returns:
        A 4D tensor representing `activation(conv2d(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    Example:

    >>> x = np.random.rand(4, 10, 10, 128)
    >>> y = keras.layers.Conv2D(32, 3, activation='relu')(x)
    >>> print(y.shape)
    (4, 8, 8, 32)
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
#mConv2D

#https://github.com/keras-team/keras/blob/master/keras/src/layers/convolutional/base_conv_transpose.py
"""Keras base class for transpose convolution layers."""

from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_output_shape,
)
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils.argument_validation import standardize_padding
from keras.src.utils.argument_validation import standardize_tuple

class mBaseConvTranspose(Layer):
    """Abstract N-D transposed convolution layer.

    The need for transposed convolutions generally arises from the desire to use
    a transformation going in the opposite direction of a normal convolution,
    i.e., from something that has the shape of the output of some convolution to
    something that has the shape of its input while maintaining a connectivity
    pattern that is compatible with said convolution.

    Args:
        rank: int, the rank of the transposed convolution, e.g. 2 for 2D
            transposed convolution.
        filters: int, the dimension of the output space (the number of filters
            in the transposed convolution).
        kernel_size: int or tuple/list of `rank` integers, specifying the size
            of the transposed convolution window.
        strides: int or tuple/list of `rank` integers, specifying the stride
            length of the transposed convolution. If only one int is specified,
            the same stride size will be used for all dimensions.
            `strides > 1` is incompatible with `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of `rank` integers, specifying the
            dilation rate to use for dilated convolution. If only one int is
            specified, the same dilation rate will be used for all dimensions.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        output_padding=None,
        data_format=None,
        dilation_rate=1,

        halfbandwidth=0, 
        param_reduction=0.5, 
        form='diagonal', 
        input2d_width='10', 
        output2d_width='10',
        window2d_width='1.41', 

        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        **kwargs,
    ):
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.rank = rank
        self.filters = filters
        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.dilation_rate = standardize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.padding = standardize_padding(padding)
        if output_padding is None:
            self.output_padding = None
        else:
            self.output_padding = standardize_tuple(
                output_padding,
                rank,
                "output_padding",
            )
        self.data_format = standardize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.data_format = self.data_format

        self.halfbandwidth = halfbandwidth
        self.form = form
        self.reduction_sv = param_reduction
        self.input2d_width = input2d_width
        self.output2d_width = output2d_width
        self.window2d_width = window2d_width
        self.num_ones = 0
        self.reduced_ratio = 0
        self.num_weights = 0
        self.reduced_ratio = 0

        if self.filters is not None and self.filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. Expected a strictly "
                f"positive value. Received filters={self.filters}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0. Received "
                f"kernel_size={self.kernel_size}."
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0. Received "
                f"strides={self.strides}."
            )

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                "`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            channel_axis = 1
            input_channel = input_shape[1]
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        kernel_shape = self.kernel_size + (
            self.filters,
            input_channel,
        )

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        #window initialization
        self.wnd = np.zeros(kernel_shape)
        self.w_corr = 1.
        #nx = input_channel // self.groups
        ny = input_channel
        nx = self.filters
        #print(self.wnd)
        #print('nxny', nx, ny)
        #print(kernel_shape)
        if self.form == 'individual':
          self.wnd = np.random.random_sample(kernel_shape)
          self.wnd = np.where(self.wnd < self.reduction_sv, 0, 1)
        elif self.form == 'kernel':
          for ix in range(nx):
            for iy in range(ny):
              if random.random() > self.reduction_sv:
                self.wnd[..., ix, iy] = 1
        elif self.form == 'diagonal':
          self.halfbandwidth = (nx*ny / math.sqrt(nx*nx + ny*ny)) * (1. - math.sqrt(self.reduction_sv)) 
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
              self.wnd[..., ix1:ix2, iy:iy+1] = 1
            #for ixiy
          else:
            self.wnd = np.ones(kernel_shape)
          #endif ny>1
        elif self.form == '2d':
          if ny > 1:
            nx1 = self.input2d_width
            nx2 = nx // self.input2d_width
            ny1 = self.output2d_width
            ny2 = ny // self.output2d_width
            d1 = self.window2d_width
            d2 = self.window2d_width * self.window2d_width
            #print('2d', nx1, nx2, ny1, ny2, d1, d2)
            #####original precise but slow version 240401
            #for ix in range(nx):
            #  for iy in range(ny):
            #    dx = (ix % nx1) / nx1 - (iy % ny1) / ny1
            #    dy = (ix // nx1) / nx2 - (iy // ny1) / ny2
            #    if (dx * dx + dy * dy < d2): 
            #      wnd[ix][iy] = 1
            #      self.num_ones += 1
            #    #endif
            #####integer version 240406
            for ix in range(nx):
              ox = (ix % nx1) / nx1
              oy = (ix // nx1) / nx2
              oymin = max(math.ceil((oy - d1) * ny2), 0)
              oymax = min(math.ceil((oy + d1) * ny2), ny2)
              for ky in range(oymin, oymax):
                dx = d2 - (ky/ny2 - oy) * (ky/ny2 - oy)
                if dx > 0:
                  dx = math.sqrt(dx)
                  oxmin = max(math.ceil((ox - dx) * ny1), 0)
                  oxmax = min(math.ceil((ox + dx) * ny1), ny1)
                  if (oxmax > oxmin) :
                    self.wnd[..., ix, (ky*ny1+oxmin):(ky*ny1+oxmax)] = 1
                    #self.num_ones += oxmax - oxmin
                  #endif
                #endif dx > 0
              #for ky
            #for ixiy
          else:
            #self.wnd[:,:] = 1
            self.wnd = np.ones(kernel_shape)
            #self.num_ones += nx
          #endif ny>1
        #endif self.form
        self.num_ones = np.sum(self.wnd)
        self.num_weights = self.wnd.size
        self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
        if self.num_ones > 0:
          self.w_corr = self.num_weights / self.num_ones
        self.kernel.assign(self.kernel * (self.wnd * self.w_corr))
        #self.window.assign(self.wnd)

        self.built = True

    def call(self, inputs):
        outputs = ops.conv_transpose(
            inputs,
            self.kernel * self.wnd,
            strides=list(self.strides),
            padding=self.padding,
            output_padding=self.output_padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs += bias

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return compute_conv_transpose_output_shape(
            input_shape,
            self.kernel_size,
            self.filters,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config

    def get_num_zeros(self):
        return(self.num_weights - self.num_ones)
    def get_num_weights(self):
        return(self.num_weights)
    def get_reduced_ratio(self):
        return(self.reduced_ratio)
    def get_halfbandwidth(self):
        return(self.halfbandwidth)
#-----

#https://github.com/keras-team/keras/blob/master/keras/src/layers/convolutional/conv2d_transpose.py
from keras.src.api_export import keras_export
#from keras.src.layers.convolutional.base_conv_transpose import BaseConvTranspose


@keras_export(
    [
        "keras.layers.mConv2DTranspose",
        "keras.layers.mConvolution2DTranspose",
    ]
)
class mConv2DTranspose(mBaseConvTranspose):
    """2D transposed convolution layer.

    The need for transposed convolutions generally arise from the desire to use
    a transformation going in the opposite direction of a normal convolution,
    i.e., from something that has the shape of the output of some convolution
    to something that has the shape of its input while maintaining a
    connectivity pattern that is compatible with said convolution.

    Args:
        filters: int, the dimension of the output space (the number of filters
            in the transposed convolution).
        kernel_size: int or tuple/list of 1 integer, specifying the size of the
            transposed convolution window.
        strides: int or tuple/list of 1 integer, specifying the stride length
            of the transposed convolution. `strides > 1` is incompatible with
            `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dilation_rate: int or tuple/list of 1 integers, specifying the dilation
            rate to use for dilated transposed convolution.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.

    Input shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, height, width, channels)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, channels, height, width)`

    Output shape:

    - If `data_format="channels_last"`:
        A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
    - If `data_format="channels_first"`:
        A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`

    Returns:
        A 4D tensor representing
        `activation(conv2d_transpose(inputs, kernel) + bias)`.

    Raises:
        ValueError: when both `strides > 1` and `dilation_rate > 1`.

    References:
    - [A guide to convolution arithmetic for deep learning](
        https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional Networks](
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)

    Example:

    >>> x = np.random.rand(4, 10, 8, 128)
    >>> y = keras.layers.Conv2DTranspose(32, 2, 2, activation='relu')(x)
    >>> print(y.shape)
    (4, 20, 16, 32)
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )


