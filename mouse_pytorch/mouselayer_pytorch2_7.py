#https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
#250713: Init routines were updated.

# mypy: allow-untyped-defs
import math
from typing import Optional, Union
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch._torch_docs import reproducibility_notes
from torch.nn import functional as F, init
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parameter import Parameter, UninitializedParameter

#from .lazy import LazyModuleMixin
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple

import math
import numpy as np

__all__ = [
    "mConv2d",
    "mConvTranspose2d",
]

convolution_notes = {
    "groups_note": r"""* :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).""",
    "depthwise_separable_note": r"""When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.""",
}  # noqa: B950


class _mConvNd(Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def _conv_forward(  # type: ignore[empty-body]
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor: ...

    in_channels: int
    _reversed_padding_repeated_twice: list[int]
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: Union[str, tuple[int, ...]]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    window: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: Union[str, tuple[int, ...]],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        
        halfbandwidth: float, 
        param_reduction: float, 
        form: str, 
        input2d_width: int, 
        output2d_width: int,
        window2d_width: float, 
        print_param_usage: bool,

        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self.halfbandwidth = halfbandwidth
        self.reduction_sv = param_reduction
        self.form = form
        self.input2d_width = input2d_width
        self.output2d_width = output2d_width
        self.window2d_width = window2d_width
        self.print_param_usage = print_param_usage
        self.num_ones = 0
        self.reduced_ratio = 0
        self.num_weights = 0
        self.reduced_ratio = 0
        
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            self.weight = Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            self.window = Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            self.window = Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"
    def get_num_zeros(self):
        return(self.num_weights - self.num_ones)
    def get_num_weights(self):
        return(self.num_weights)
    def get_reduced_ratio(self):
        return(self.reduced_ratio)
    def get_halfbandwidth(self):
        return(self.halfbandwidth)

class mConv2d(_mConvNd):
    __doc__ = (
        r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """
        + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or an int / a tuple of ints giving the
      amount of implicit padding applied on both sides.
"""
        """
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the \u00e0 trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
"""
        r"""

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        {depthwise_separable_note}

    Note:
        {cudnn_reproducibility_note}

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    """.format(**reproducibility_notes, **convolution_notes)
        + r"""

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type

        halfbandwidth=0, 
        param_reduction=0.5, 
        form='diagonal', 
        input2d_width=10, 
        output2d_width=10,
        window2d_width=1.41, 
        print_param_usage=False,

        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            
            halfbandwidth, 
            param_reduction, 
            form, 
            input2d_width, 
            output2d_width,
            window2d_width, 
            print_param_usage, 

            **factory_kwargs,
        )
        #window initialization
        kernel_shape = (out_channels, in_channels // groups) + kernel_size_
        wnd =np.zeros(kernel_shape)
        self.w_corr = 1.
        nx = in_channels // groups
        ny = self.out_channels
        if self.form == 'individual':
          wnd = np.random.random_sample(kernel_shape)
          wnd = np.where(wnd < self.reduction_sv, 0, 1)
        elif self.form == 'kernel':
          for ix in range(nx):
            for iy in range(ny):
              if random.random() > self.reduction_sv:
                #wnd[..., ix, iy] = 1
                wnd[iy, ix, ...] = 1
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
              #wnd[..., ix1:ix2, iy:iy+1] = 1
              wnd[iy:iy+1, ix1:ix2, ...] = 1
            #for ixiy
          else:
            wnd = np.ones(kernel_shape)
          #endif ny>1
        elif self.form == '2d':
          if ny > 1:
            nx1 = self.input2d_width
            nx2 = nx // self.input2d_width
            ny1 = self.output2d_width
            ny2 = ny // self.output2d_width
            d1 = self.window2d_width
            d2 = self.window2d_width * self.window2d_width
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
                    #wnd[..., ix, (ky*ny1+oxmin):(ky*ny1+oxmax)] = 1
                    wnd[(ky*ny1+oxmin):(ky*ny1+oxmax), ix, ...] = 1
                  #endif
                #endif dx > 0
              #for ky
            #for ixiy
          else:
            wnd = np.ones(kernel_shape)
          #endif ny>1
        #endif self.form
        self.num_ones = np.sum(wnd)
        self.num_weights = wnd.size
        self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
        if self.num_ones > 0:
          self.w_corr = self.num_weights / self.num_ones
        if (self.print_param_usage): print ("param %usage:", 100.0/self.w_corr)
        self.window = Parameter(torch.Tensor(wnd))
        self.weight = Parameter(self.weight * torch.Tensor(wnd * self.w_corr))        

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight * self.window, self.bias)

class _mConvTransposeNd(_mConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        
        halfbandwidth, 
        param_reduction, 
        form, 
        input2d_width, 
        output2d_width,
        window2d_width, 
        print_param_usage,

        device=None,
        dtype=None,
    ) -> None:
        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
                    
            halfbandwidth, 
            param_reduction, 
            form, 
            input2d_width, 
            output2d_width,
            window2d_width, 
            print_param_usage,

            **factory_kwargs,
        )

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(
        self,
        input: Tensor,
        output_size: Optional[list[int]],
        stride: list[int],
        padding: list[int],
        kernel_size: list[int],
        num_spatial_dims: int,
        dilation: Optional[list[int]] = None,
    ) -> list[int]:
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    f"ConvTranspose{num_spatial_dims}D: for {input.dim()}D input, output_size must have {num_spatial_dims} "
                    f"or {num_non_spatial_dims + num_spatial_dims} elements (got {len(output_size)})"
                )

            min_sizes = torch.jit.annotate(list[int], [])
            max_sizes = torch.jit.annotate(list[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range "
                        f"from {min_sizes} to {max_sizes} (for an input of {input.size()[2:]})"
                    )

            res = torch.jit.annotate(list[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def get_num_zeros(self):
        return(self.num_weights - self.num_ones)
    def get_num_weights(self):
        return(self.num_weights)
    def get_reduced_ratio(self):
        return(self.reduced_ratio)
    def get_halfbandwidth(self):
        return(self.halfbandwidth)

class mConvTranspose2d(_mConvTransposeNd):
    __doc__ = (
        r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation. When stride > 1, ConvTranspose2d inserts zeros between input
      elements along the spatial dimensions before applying the convolution kernel. This zero-insertion operation is the standard
      behavior of transposed convolutions, which can increase the spatial resolution and is equivalent to a learnable
      upsampling operation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.
"""
        """
    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.
"""
        r"""
    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Note:
        {cudnn_reproducibility_note}

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    """.format(**reproducibility_notes, **convolution_notes)
        + r"""

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

        .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
        .. math::
              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        
        halfbandwidth=0, 
        param_reduction=0.5, 
        form='diagonal', 
        input2d_width=10, 
        output2d_width=10,
        window2d_width=1.41, 
        print_param_usage=False,

        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            
            halfbandwidth, 
            param_reduction, 
            form, 
            input2d_width, 
            output2d_width,
            window2d_width, 
            print_param_usage, 

            **factory_kwargs,
        )
        #window initialization
        kernel_shape = (in_channels, out_channels // groups) + kernel_size
        wnd =np.zeros(kernel_shape)
        self.w_corr = 1.
        nx = out_channels // groups
        ny = self.in_channels
        if self.form == 'individual':
          wnd = np.random.random_sample(kernel_shape)
          wnd = np.where(wnd < self.reduction_sv, 0, 1)
        elif self.form == 'kernel':
          for ix in range(nx):
            for iy in range(ny):
              if random.random() > self.reduction_sv:
                #wnd[..., ix, iy] = 1
                wnd[iy, ix, ...] = 1
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
              #wnd[..., ix1:ix2, iy:iy+1] = 1
              wnd[iy:iy+1, ix1:ix2, ...] = 1
            #for ixiy
          else:
            wnd = np.ones(kernel_shape)
          #endif ny>1
        elif self.form == '2d':
          if ny > 1:
            nx1 = self.input2d_width
            nx2 = nx // self.input2d_width
            ny1 = self.output2d_width
            ny2 = ny // self.output2d_width
            d1 = self.window2d_width
            d2 = self.window2d_width * self.window2d_width
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
                    #wnd[..., ix, (ky*ny1+oxmin):(ky*ny1+oxmax)] = 1
                    wnd[(ky*ny1+oxmin):(ky*ny1+oxmax), ix, ...] = 1
                  #endif
                #endif dx > 0
              #for ky
            #for ixiy
          else:
            wnd = np.ones(kernel_shape)
          #endif ny>1
        #endif self.form
        self.num_ones = np.sum(wnd)
        self.num_weights = wnd.size
        self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
        if self.num_ones > 0:
          self.w_corr = self.num_weights / self.num_ones
        if (self.print_param_usage): print ("param %usage:", 100.0/self.w_corr)
        self.window = Parameter(torch.Tensor(wnd))
        self.weight = Parameter(self.weight * torch.Tensor(wnd * self.w_corr))        

    def forward(self, input: Tensor, output_size: Optional[list[int]] = None) -> Tensor:
        """
        Performs the forward pass.

        Attributes:
            input (Tensor): The input tensor.
            output_size (list[int], optional): A list of integers representing
                the size of the output tensor. Default is None.
        """
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )

        return F.conv_transpose2d(
            input,
            self.weight * self.window,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )