
import numpy as np
import torch
import torch.nn.functional as F
from torch import _VF


from typing import List
from functools import reduce
from ...ppq_.ppq.core.data import convert_any_to_torch_tensor
from ppq.IR import Operation
from ppq.executor.op import TorchBackendContext
from ppq.utils import process_attribute
from ppq.core import DataType, TargetPlatform, convert_any_to_python_primary_type, LSTM_FLATTEN_WEIGHT_ATTRIB
from ppq.executor.op.torch.default import convert_onnx_pads_to_torch
from ppq.executor.op.torch.base import (ASSERT_NUM_OF_INPUT, GET_ATTRIBUTE_FROM_OPERATION, GET_VALUE_FROM_INPUTS, 
                                        VALUE_TO_EXECUTING_DEVICE)

from maca_quantizer.utils.utils import maca_error, maca_warning
from .op_utils import  last_index_impl, _arbitrary_dim_shift_and_insert_zero




def ArgMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input_value] = values
    dim = op.attributes.get('axis', 0)
    keepdim = bool(op.attributes.get('keepdims', 1))                    # default 1 means keep reduced dimension
    select_last_index = bool(op.attributes.get('select_last_index', 0)) # default is False (first index)
    if not select_last_index:
        output = torch.argmax(input_value, dim=dim, keepdim=keepdim)
    else:
        # implement select_last_index=1
        output = last_index_impl(op, input_value, axis=dim, keep_dim=keepdim)
    return output


def ArgMin_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input_value] = values
    dim = op.attributes.get('axis', 0)
    keepdim = bool(op.attributes.get('keepdims', 1))
    select_last_index = bool(op.attributes.get('select_last_index', 0)) # default is False (first index)
    if not select_last_index:
        output = torch.argmin(input_value, dim=dim, keepdim=keepdim)
    else:
        # implement select_last_index=1
        output = last_index_impl(op, input_value, axis=dim, keep_dim=keepdim)
    return output


def Ceil_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    input_data = values[0]
    output = torch.ceil(input_data)
    return output


def CumSum_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    data, axis = values
    exclusive = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='exclusive', default=0))
    reverse   = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='reverse', default=0))

    if reverse:
        data = torch.flip(data, dims=[axis])
    if exclusive:
        data = _arbitrary_dim_shift_and_insert_zero(data, insert_dim=axis)
    output = torch.cumsum(data, axis)

    if reverse:
        output = torch.flip(output, dims=[axis])

    return output


def DepthToSpace_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    # SubpixelUp in caffe
    input_data = values[0]
    upsample = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='blocksize', default=1)
    mode     = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='mode', default='DCR')
    b,c,h,w = input_data.shape
    if mode == 'DCR':
        output = torch.reshape(input_data, [b, upsample, upsample, c//(upsample**2),  h, w])
        output = torch.permute(output, [0, 3, 4, 1, 5, 2])
        output = torch.reshape(output, [b, c // (upsample ** 2), h * upsample, w * upsample])
    else:  # mode == 'CRD' is correct.
        # output = F.pixel_shuffle(input_data, upsample)
        output = torch.reshape(input_data, [b, c//(upsample**2),  upsample, upsample, h, w])
        output = torch.permute(output, [0, 1, 4, 2, 5, 3])
        output = torch.reshape(output, [b, c // (upsample ** 2), h * upsample, w * upsample])
    return output


def Einsum_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Attributes
        equation : string (required)
        Einsum expression string.
    Inputs (1 - ∞)
        Inputs (variadic, differentiable) : T
        Operands
    Outputs
        Output (differentiable) : T
        Output tensor
    """
    equation = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='equation', compulsive=True)
    output = torch.einsum(equation, values)
    return output



def AveragePool_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    process_attribute(op.attributes, values[0].shape[2:])

    [x] = values
    onnx_pads    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    stride       = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    ceil_mode    = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=0))
    count_include_pad = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute="count_include_pad", default=0))
    if op.type   == 'GlobalAveragePool': kernel_size = x.size()[2:]
    else: kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)

    ndim = x.ndim
    # pool 1d
    if ndim == 3:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) != 1:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.avg_pool1d(
            x, kernel_size=kernel_size, padding=torch_pads,
            stride=stride, ceil_mode=ceil_mode, 
            count_include_pad=count_include_pad)

    # pool 2d
    elif ndim == 4:
        # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            # torch does not support padding contains 4 value, there is a fix of it.
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
                onnx_pads = 0

        output = F.avg_pool2d(
            x, kernel_size=kernel_size,
            padding=onnx_pads, stride=stride, ceil_mode=ceil_mode, 
            count_include_pad=count_include_pad)

    # pool 3d
    elif ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) != 3:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.avg_pool3d(
            x, kernel_size=kernel_size, padding=torch_pads,
            stride=stride, ceil_mode=ceil_mode,
            count_include_pad=count_include_pad)

    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')

    return output


def MaxUnpool_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    """
    Attributes
        kernel_shape : list of ints (required)
            The size of the kernel along each axis.

        pads : list of ints
            Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. 

        strides : list of ints
            Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
    
    Inputs (2 - 3)
        X (differentiable) : T1
            Input data tensor that has to be unpooled. 

        I (non-differentiable) : T2
            Input data tensor containing the indices corresponding to elements in the first input tensor X.

        output_shape (optional, non-differentiable) : T2
            The shape of the output can be explicitly set which will cause pads values to be auto generated.
            If 'output_shape' is specified, 'pads' values are ignored.
    
    Outputs
        output (differentiable) : T1
            Output data tensor that contains the result of the unpooling.
    """
    
    
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    kernel_shape = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)
    strides      = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    onnx_pads    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)

    x, indices = values[:2]
    output_shape = values[2].tolist() if len(values)>2 else None
    ndim = x.ndim
    convert_any_to_python_primary_type(output_shape)
    # If 'output_shape' is specified, 'pads' values are ignored.
    onnx_pads = 0 if output_shape is not None else onnx_pads

    if ndim==3:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) != 1:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        unpool_obj = torch.nn.MaxUnpool1d(kernel_shape, strides, padding=torch_pads)

    elif ndim==4:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(onnx_pads, list) and len(onnx_pads) != 2:
            x = F.pad(x, pad=torch_pads)
            torch_pads = 0
        unpool_obj = torch.nn.MaxUnpool2d(kernel_shape, strides, padding=torch_pads)

    elif ndim==5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) != 3:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        unpool_obj = torch.nn.MaxUnpool3d(kernel_shape, strides, padding=torch_pads)

    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')

    output = unpool_obj(x, indices, output_shape)
    
    return output


def ReduceMin_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if len(input_value) == 0:
        output = input_value
    else:
        if dim is None:
            #  The default is to reduce over all the dimensions of the input tensor
            output = torch.min(input_value)
            if keepdim:
                output = output.reshape([1] * input_value.dim())
        else:
            output, _ = torch.min(input_value, dim=dim[0], keepdim=keepdim)
    return output


def ReduceProd_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if len(input_value) == 0:
        output = input_value
    else:
        if dim is None:
            #  The default is to reduce over all the dimensions of the input tensor
            output = torch.prod(input_value)
            if keepdim:
                output = output.reshape([1] * input_value.dim())
        else:
            output = torch.prod(input_value, dim=dim[0], keepdim=keepdim)

    return output

def Round_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [x] = values
    output = torch.round(x)
    return output


def ScatterElements_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    """
    ScatterElements takes three inputs data, updates, 
    and indices of the same rank r >= 1 and an optional attribute axis that identifies an axis of data 
    (by default, the outer-most axis, that is axis 0). 
    
    The output of the operation is produced by creating a copy of the input data, 
    and then updating its value to values specified by updates at specific index positions specified by indices. 
    Its output shape is the same as the shape of data.

    For each entry in updates, 
    the target index in data is obtained by combining the corresponding entry in indices with the index of the entry itself: 
        the index-value for dimension = axis is obtained from the value of the corresponding entry in indices and the index-value 
    for dimension != axis is obtained from the index of the entry itself.

    reduction allows specification of an optional reduction operation, 
        which is applied to all values in updates tensor into output at the specified indices.
    In cases where reduction is set to "none", indices should not have duplicate entries: that is, 
        if idx1 != idx2, then indices[idx1] != indices[idx2]. 
    
    For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] = updates[i][j] if axis = 0,
    output[i][indices[i][j]] = updates[i][j] if axis = 1,
    When reduction is set to "add", the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] += updates[i][j] if axis = 0,
    output[i][indices[i][j]] += updates[i][j] if axis = 1,
    When reduction is set to "mul", the update corresponding to the [i][j] entry is performed as below:

    output[indices[i][j]][j] *= updates[i][j] if axis = 0,
    output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

    Attributes
        axis : int (default is 0)
            Which axis to scatter on. Negative value means counting dimensions from the back.
                Accepted range is [-r, r-1] where r = rank(data).
            reduction : string (default is none)
            Type of reduction to apply: none (default), add, mul. 

            'none': no reduction applied. 
            'add': reduction using the addition operation. 
            'mul': reduction using the multiplication operation.

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of r >= 1 (same rank as input). 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

        updates (differentiable) : T
            Tensor of rank r >=1 (same rank and shape as indices)

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1 (same rank as input).
    """
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    value, indices, updates = values

    dim = op.attributes.get('axis', 0)
    reduction = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='reduction', default='none')
    attr_map = {
        'add' : 'add',
        'mul' : 'multiply'
    }
    # Negative indices
    indices[indices < 0] += value.shape[dim]
    if reduction=='none':
        output = value.scatter(dim, indices, updates)
    else:
        output = value.scatter(dim, indices, updates, reduce=attr_map[reduction])
    return output



def SequenceAt_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_squeece, position] = values
    position = convert_any_to_python_primary_type(position)
    output = input_squeece[position]
    return output


def SplitToSequence_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Attributes
        axis : int (default is 0)
            Which axis to split on. Accepted range is [-rank, rank-1].
        keepdims : int (default is 1)
            Keep the split dimension or not. Default 1.
    Inputs (1 - 2)
        input : T
            The tensor to split
        split (optional) : I
            Length of each output. It can be either a scalar(tensor of empty shape), or a 1-D tensor. All values must be >= 0.
    Outputs
        output_sequence : S
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=0)
    keepdim = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='keepdims', default=1))
    data = values[0]
    split = values[1] if len(values)>1 else None
    if split is None:
        block_num = data.shape[axis]
        output = torch.tensor_split(data, sections=block_num, dim=axis)
    else:
        if split.numel()==1:
            block_num = data.shape[axis]//split
            output = torch.tensor_split(data, block_num, dim=axis)
        else:
            split_tensors = torch.tensor_split(data, split, dim=axis)
            split_indices = torch.cumsum(split,dim=0)
            output = []
            start_idx = 0
            for i in split_indices:
                tensors = torch.concat(split_tensors[start_idx:i], dim=axis)
                output.append(tensors)
                start_idx = i

    if not keepdim:
        output = [t.squeeze(dim=axis) for t in output]
    return output


def Xor_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    output = torch.logical_xor(a, b)
    return output


# ======================================================== split ========================================================

# TODO: shape might contain 0, needs better solution
def Reshape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Reshape the input tensor similar to numpy.reshape. First input is the
    data tensor, second input is a shape tensor which specifies the output
    shape. It outputs the reshaped tensor.

    At most one dimension of the new shape can be -1.
    In this case, the value is inferred from the size of the tensor and the remaining dimensions.
    A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
    If 'allowzero' is set, and the new shape includes 0,
        the dimension will be set explicitly to zero (i.e. not taken from input tensor)

    Attributes
        allowzero : int (default is 0)
        (Optional) By default, when any value in the 'shape' input is equal to zero
            the corresponding dimension value is copied from the input tensor dynamically.

        allowzero=1 indicates that if any value in the 'shape' input is set to zero,
            the zero value is honored, similar to NumPy.
    Inputs
        data (differentiable) : T
            An input tensor.

        shape (non-differentiable) : tensor(int64)
            Specified shape for output.

    Outputs
        reshaped (differentiable) : T
            Reshaped data.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    allowzero = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='allowzero', default=0))
    # if allowzero: raise NotImplemented(f'Not implemented yet for allowzero={allowzero}.')
    if not allowzero and 'allowzero' in op.attributes:
        op.attributes.pop('allowzero')
    data, shape = values
    shape = shape.cpu()

    # Avoid the element 0 in the shape that comes with onnx, such as [0,-1]
    shape = [shape[i] if shape[i] != 0 else data.shape[i] for i in range(len(shape))]

    # If the element in shape is a tensor, convert it to a value, for example,
    # convert [1, tensor(-1)] to [1, -1]
    shape = [shape[i].item() if hasattr(shape[i], 'item') else shape[i] for i in range(len(shape))]
    if shape[0] == -1:
        shape_ = shape
    else:
        shape_ = [data.shape[0]] + shape[1:]
    try:
        output = data.reshape(shape_)
    except RuntimeError:
        output = data.reshape(shape)
    return output



def Resize_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    # Refs: ppq/executor/op/torch/default.py
    """
    value = values[0]
    # Not used roi
    # roi  = input_value[1] if len(input_value) > 1 else None
    scale_factor = values[2].cpu() if len(values) > 2 else None
    size = values[-1].cpu().tolist() if (len(values) == 4 and values[-1] != None) else None
    mode = op.attributes.get('mode', 'nearest')
    if mode == 'cubic':
        mode = 'bicubic'
    # onnx resize 'linear' model include N-linear interpolate for N-D tensor
    linear_mode_map = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}

    # If 'size' is specified, then set scales to empty data (zero shape) in this operator's input list.
    if size is None or len(size) == 0:
        size = None
        if scale_factor.numel() == 1:
            scale_factor = scale_factor.item()
        else:
            scale_factor = scale_factor.tolist()
            if len(scale_factor) == 2:
                # 大家相安无事，和平共处
                pass
            elif len(scale_factor) == 4 or len(scale_factor) == 5:
                if scale_factor[:2] != [1, 1]:
                    raise NotImplementedError(
                        'Can not resize your image with current op, '
                        'cause 4-dimension resize is not implemented with pytorch.')
                scale_factor = scale_factor[2:]
            else:
                raise NotImplementedError(
                    'Can not resize your image with current op, '
                    f'cause {len(scale_factor)}-dimension resize is not implemented with pytorch.')
        size = list(value.shape[:2]) + [round(value.shape[2 + i] * scale_factor[i]) for i in range(len(scale_factor))]
        scale_factor = None

    # the sizes in onnx is 4-D while in pytorch is 2-D
    # check the dim.0 & dim.1 is equal, then remain dim.2 and dim.3
    # avoid batchsize causes error
    if size[1] != list(value.shape)[1]:
        maca_error(f"Data shape \'{size}\' and Resize size attribute \'{list(value.shape)}\' "
                    "dim.0 & dim.1 is not equal")
    size = size[2:]
    mode = linear_mode_map[len(size)] if mode == 'linear' else mode

    if mode == 'cubic':
        maca_warning('Only support bicubic now')
        assert (len(size[2:]) == 2)
        mode = 'bicubic'

    # PATCH 2022.04.22
    # ONNX DO NOT HAVE BILINEAR MODE, FOR 4D INPUT, WE OVERRIDE MODE TO BILINEAR
    if len(value.shape) == 4 and mode == 'linear':
        mode = 'bilinear'

    trans_mode = op.attributes.get(
        'coordinate_transformation_mode', 'half_pixel')
    if trans_mode == 'align_corners':
        output = F.interpolate(value, size, mode=mode, align_corners=True)
    else:
        output = F.interpolate(value, size, mode=mode)
    return output




def LSTM_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Computes an one-layer LSTM. This operator is usually supported via some
    custom implementation such as CuDNN.

    只支持 pytorch 导出来的 LSTM 啊亲; 必须要 7 个输入 Variable

    Computes an one-layer LSTM. This operator is usually supported via some custom implementation such as CuDNN.

    Notations:

    X - input tensor

    i - input gate

    o - output gate

    f - forget gate

    c - cell gate

    t - time step (t-1 means previous time step)

    W[iofc] - W parameter weight matrix for input, output, forget, and cell gates

    R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates

    Wb[iofc] - W bias vectors for input, output, forget, and cell gates

    Rb[iofc] - R bias vectors for input, output, forget, and cell gates

    P[iof] - P peephole weight vector for input, output, and forget gates

    WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates

    RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates

    WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates

    RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates

    PB[iof] - P peephole weight vector for backward input, output, and forget gates

    H - Hidden state

    num_directions - 2 if direction == bidirectional else 1

    Activation functions:

    Relu(x)                - max(0, x)

    Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

    Sigmoid(x)             - 1/(1 + e^{-x})

    (NOTE: Below are optional)

    Affine(x)              - alpha*x + beta

    LeakyRelu(x)           - x if x >= 0 else alpha * x

    ThresholdedRelu(x)     - x if x >= alpha else 0

    ScaledTanh(x)          - alpha*Tanh(beta*x)

    HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

    Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

    Softsign(x)            - x/(1 + |x|)

    Softplus(x)            - log(1 + e^x)
    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

    - Ct = ft (.) Ct-1 + it (.) ct

    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

    - Ht = ot (.) h(Ct)
    This operator has optional inputs/outputs. 
    See the doc for more details about the representation of optional arguments. 
    An empty string may be used in the place of an actual argument's name to indicate a missing argument. 
    Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

    Version
    This version of the operator has been available since version 7 of the default ONNX operator set.

    Attributes
        activation_alpha : list of floats
            Optional scaling values used by some activation functions. 
            The values are consumed in the order of activation functions, 
                for example (f, g, h) in LSTM. 
    
            Default values are the same as of corresponding ONNX operators.For example with LeakyRelu, the default alpha is 0.01.
    
        activation_beta : list of floats
            Optional scaling values used by some activation functions. 
            The values are consumed in the order of activation functions, 
            for example (f, g, h) in LSTM. 
            
            Default values are the same as of corresponding ONNX operators.
    
        activations : list of strings
            A list of 3 (or 6 if bidirectional) activation functions for input, 
            output, forget, cell, and hidden. 
            
            The activation functions must be one of the activation functions specified above. 
            Optional: See the equations for default if not specified.
    
        clip : float
            Cell clip threshold. Clipping bounds the elements of a tensor in the range of 
            [-threshold, +threshold] and is applied to the input of activations.
            No clip if not specified.
        
        direction : string (default is forward)
            Specify if the RNN is forward, reverse, or bidirectional. 
            Must be one of forward (default), reverse, or bidirectional.

        hidden_size : int
            Number of neurons in the hidden layer
    
        input_forget : int (default is 0)
            Couple the input and forget gates if 1.
    
    Inputs (3 - 8)
        X : T
            The input sequences packed (and potentially padded) into one 3-D tensor 
                with the shape of `[seq_length, batch_size, input_size]`.
   
        W : T
            The weight tensor for the gates. Concatenation of `W[iofc]` and `WB[iofc]` 
            (if bidirectional) along dimension 0. The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
    
        R : T
            The recurrence weight tensor. Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0. 
            This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
    
        B (optional) : T
            The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, 
            and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. 
            
            This tensor has shape `[num_directions, 8*hidden_size]`. 
            Optional: If not specified - assumed to be 0.
    
        sequence_lens (optional) : T1
            Optional tensor specifying lengths of the sequences in a batch. 
            If not specified - assumed all sequences in the batch to have length `seq_length`. 
            It has shape `[batch_size]`.
        
        initial_h (optional) : T
            Optional initial value of the hidden. 
            If not specified - assumed to be 0. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        initial_c (optional) : T
            Optional initial value of the cell. 
            If not specified - assumed to be 0. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        P (optional) : T
            The weight tensor for peepholes.
            Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0. 
            It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
    
    Outputs (0 - 3)
        Y (optional) : T
            A tensor that concats all the intermediate output values of the hidden. 
            It has shape `[seq_length, num_directions, batch_size, hidden_size]`.
    
        Y_h (optional) : T
            The last output value of the hidden. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        Y_c (optional) : T
            The last output value of the cell. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=8)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)

    # check attributes
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations      = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip             = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction        = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size      = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    input_forget     = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='input_forget', default=0)
    layout           = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    if layout != 0: raise NotImplementedError('PPQ do not support LSTM with layout != 1.')
    if activation_alpha is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activation_beta is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activations is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')


    # first 3 are mandatory input
    #  input tensor with shape [seq_length, batch_size, input_size]
    #  weights tensor with shape [4*hidden_size, input_size]
    #  recurrence tensor with shape [4*hidden_size, hidden_size]
    x, w, r   = values[: 3]
    b         = GET_VALUE_FROM_INPUTS(values, 3)    # optional bias tensor with shape [8*hidden_size]
    seq_len   = GET_VALUE_FROM_INPUTS(values, 4)    # optional tensor specifying sequence lengths in a batch, shape: [batch_size]
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)    # optional initial H, shape: [num_directions, batch_size, hidden_size]
    initial_c = GET_VALUE_FROM_INPUTS(values, 6)    # optional initial C, shape: [num_directions, batch_size, hidden_size]
    p         = GET_VALUE_FROM_INPUTS(values, 7)
    if p is not None: raise NotImplementedError('PPQ do not support LSTM with peepholes.')
    
    # sequence length will be dropped without warrning.
    # if seq_len is not None: raise NotImplementedError('PPQ do not support LSTM with explicite length.')
    
    # fix seq_len dtype
    if op.inputs[4].dtype not in {DataType.INT32}:
        op.inputs[4].dtype = DataType.INT32
        if seq_len is not None and op.inputs[4].is_parameter:
            op.inputs[4].value = convert_any_to_torch_tensor(op.inputs[4].value, dtype=torch.int32)
    
    # flag
    bidirectional = (direction == 'bidirectional')
    has_bias      = (b is not None)

    if direction == 'reverse': raise NotImplementedError('GRU do not support reverse mode now.')
    
    # create flatten weights:
    if LSTM_FLATTEN_WEIGHT_ATTRIB not in op.attributes:
        forward_w = torch.cat([
            w[0][hidden_size * 0: hidden_size * 1],
            w[0][hidden_size * 2: hidden_size * 3],
            w[0][hidden_size * 3: hidden_size * 4],
            w[0][hidden_size * 1: hidden_size * 2]], dim=0).contiguous()
        forward_r = torch.cat([
            r[0][hidden_size * 0: hidden_size * 1],
            r[0][hidden_size * 2: hidden_size * 3],
            r[0][hidden_size * 3: hidden_size * 4],
            r[0][hidden_size * 1: hidden_size * 2]], dim=0).contiguous()
        if has_bias:
            forward_bias_1 = torch.cat([
                b[0, hidden_size * 0: hidden_size * 1],
                b[0, hidden_size * 2: hidden_size * 3],
                b[0, hidden_size * 3: hidden_size * 4],
                b[0, hidden_size * 1: hidden_size * 2]]).contiguous()
            forward_bias_2 = torch.cat([
                b[0, hidden_size * 4: hidden_size * 5],
                b[0, hidden_size * 6: hidden_size * 7],
                b[0, hidden_size * 7: hidden_size * 8],
                b[0, hidden_size * 5: hidden_size * 6]]).contiguous()
        if bidirectional == True:
            reverse_w = torch.cat([
                w[1][hidden_size * 0: hidden_size * 1],
                w[1][hidden_size * 2: hidden_size * 3],
                w[1][hidden_size * 3: hidden_size * 4],
                w[1][hidden_size * 1: hidden_size * 2]], dim=0).contiguous()
            reverse_r = torch.cat([
                r[1][hidden_size * 0: hidden_size * 1],
                r[1][hidden_size * 2: hidden_size * 3],
                r[1][hidden_size * 3: hidden_size * 4],
                r[1][hidden_size * 1: hidden_size * 2]], dim=0).contiguous()
            if has_bias:
                reverse_bias_1 = torch.cat([
                    b[1, hidden_size * 0: hidden_size * 1],
                    b[1, hidden_size * 2: hidden_size * 3],
                    b[1, hidden_size * 3: hidden_size * 4],
                    b[1, hidden_size * 1: hidden_size * 2]]).contiguous()
                reverse_bias_2 = torch.cat([
                    b[1, hidden_size * 4: hidden_size * 5],
                    b[1, hidden_size * 6: hidden_size * 7],
                    b[1, hidden_size * 7: hidden_size * 8],
                    b[1, hidden_size * 5: hidden_size * 6]]).contiguous()
        
        flatten_weight = [forward_w, forward_r]
        if has_bias:                   flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2]
        if bidirectional:              flatten_weight = [forward_w, forward_r, reverse_w, reverse_r]
        if bidirectional and has_bias: flatten_weight = [
            forward_w, forward_r, forward_bias_1, forward_bias_2, reverse_w, reverse_r, reverse_bias_1, reverse_bias_2]
        op.set_extension_attrib(LSTM_FLATTEN_WEIGHT_ATTRIB, flatten_weight)
    # end if
    
    s = 2 if bidirectional else 1
    if initial_h is None:
        initial_h = torch.zeros(
            size=[s, x.shape[1], hidden_size], 
            device=x.device, dtype=torch.float32)

    if initial_c is None:
        initial_c = torch.zeros(
            size=[s, x.shape[1], hidden_size], 
            device=x.device, dtype=torch.float32)

    result = _VF.lstm(
        x.contiguous(),                                   # x
        (initial_h.contiguous(), initial_c.contiguous()), # initial hidden state
        op._detail[LSTM_FLATTEN_WEIGHT_ATTRIB],  # flatten weights
        has_bias,                                # has bias
        1,                                       # num of layer
        0.0,                                     # dropout
        False,                                   # training flag
        bidirectional,                           # bidirectional
        False)                                   # batch first

    hs, h, c = result
    if bidirectional:
        hs = hs.reshape((hs.shape[0], hs.shape[1], 2, hs.shape[-1] // 2))
        hs = hs.permute((0, 2, 1, 3))
    else:
        hs = hs.reshape((hs.shape[0], hs.shape[1], 1, hs.shape[-1]))
        hs = hs.permute((0, 2, 1, 3))
    return hs, h, c



def GroupNormalization_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    A GroupNormalization function. Carries out group normalization as described in the paper https://arxiv.org/abs/1803.08494
    This operator transforms input according to
        y = scale * (x - mean) / sqrt(variance + epsilon) + bias,

    where the mean and variance are computed per instance per group of channels, and scale and bias should be specified for each group of channels. 
    The number of groups num_groups should be divisible by the number of channels so that there are an equal number of channels per group.

    Attributes
        epsilon : float (default is 1e-05)

        num_groups : int (required)

    Inputs
        X (differentiable) : T
            Input data tensor. Dimensions for image cases are `(N x C x H x W)`, where `N` is the batch size, `C` is the number of channels, 
            and `H` and `W` are the height and width of the data. Statistics are computed for every group of channels over `C`, `H`, and `W`. 
            For non-image cases, the dimensions are in the form of `(N x C x D1 x D2 ... Dn)`.
        scale (differentiable) : T
            Scale tensor of shape `(num_groups)`.
        bias (differentiable) : T
            Bias tensor of shape `(num_groups)`.
    Outputs
        Y (differentiable) : T

    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)

    eps = GET_ATTRIBUTE_FROM_OPERATION(op, 'epsilon', default=1e-5)
    num_groups = GET_ATTRIBUTE_FROM_OPERATION(op, 'num_groups', compulsive=True)
    stash_type = GET_ATTRIBUTE_FROM_OPERATION(op, 'stash_type ', default=1)

    input_data, weight, bias = values
    # weight_ = weight.repeat(num_groups)
    # bias_   = bias.repeat(num_groups)

    output = F.group_norm(input_data, num_groups=num_groups, weight=weight, bias=bias, eps=eps)
    
    return output
