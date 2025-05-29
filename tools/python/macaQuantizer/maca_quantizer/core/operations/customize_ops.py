from sympy import re
import torch
import torch.nn.functional as F
from torch import _VF

from typing import Any, List
from functools import reduce
from ppq.IR import Operation
from ppq.core import DataType, TargetPlatform, convert_any_to_python_primary_type
from ppq.executor.op.torch.base import ASSERT_NUM_OF_INPUT, GET_ATTRIBUTE_FROM_OPERATION, VALUE_TO_EXECUTING_DEVICE
from ppq.executor.op import TorchBackendContext
from ppq.executor.op import DEFAULT_BACKEND_TABLE



def Mish_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """The operator computes the normalized exponential values for the given
    input:

    Mish(input) = input * Tanh(Softplus(input))

    The output tensor has the same shape and contains the Softmax values of the corresponding input.

    Inputs
        input (differentiable) : T

    Outputs
        output (differentiable) : T
        The output values with the same shape as the input tensor.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    # TODO :PROCESS DOMIAN
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input] = values
    # output = input * F.tanh(F.softplus(input))  # torch version < 1.2.0
    output = F.mish(input)
    return output


def Swish_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    beta = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='beta', default=1.0)
    [input_value] = values
    return input_value * torch.sigmoid(beta * input_value)



def ConvActEleActFused_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """

    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=4)
    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)

    act1_type = GET_ATTRIBUTE_FROM_OPERATION(op, 'act1_type', compulsive=True)
    act1_param_data = GET_ATTRIBUTE_FROM_OPERATION(op, 'act1_param_data', default=None)
    ele_type  = GET_ATTRIBUTE_FROM_OPERATION(op, 'ele_type',  compulsive=True)

    act2_type = GET_ATTRIBUTE_FROM_OPERATION(op, 'act2_type', default=None)
    act2_param_data = GET_ATTRIBUTE_FROM_OPERATION(op, 'act1_param_data', default=None)

    conv_output = DEFAULT_BACKEND_TABLE["Conv"](op, values[:-1], ctx)

    act1_output = _activation_forward(conv_output, act1_type, act1_param_data)


    if ele_type=='Add':
        elementwise_output = act1_output + values[-1]
    elif ele_type=='Mul':
        elementwise_output = act1_output * values[-1]
    
    if act2_type is None:
        return elementwise_output
    else:
        return _activation_forward(elementwise_output, act2_type, act2_param_data)



def _activation_forward(input, act1_type, act1_param_data):
    if act1_type == 'Relu':
        act1_output = F.relu(input)
    elif act1_type == 'Clip':
        if act1_param_data is None:
            min = float('-inf')
            max = float('+inf')
        else:
            min, max = act1_param_data
        act1_output = torch.clamp(input, min, max)
    elif act1_type == 'PRelu':
        slope = act1_param_data
        weight = slope.squeeze()
        act1_output = F.prelu(input, weight)
    elif act1_type == 'LeakyRelu':
        alpha = act1_param_data
        act1_output = F.leaky_relu(input, alpha)
    elif act1_type == 'Sigmoid':
        act1_output = torch.sigmoid(input)
    elif act1_type == 'Swish':
        act1_output = input * torch.sigmoid(1.0 * input)
    elif act1_type == 'Mish':
        act1_output = F.mish(input)
    elif act1_type == 'Softplus':
        act1_output = F.softplus(input)
    elif act1_type == 'HardSigmoid':
        alpha,beta  = 0.2 , 0.5
        if act1_param_data is not None:
            alpha,beta  = act1_param_data
        tmp = alpha * input + beta
        act1_output = torch.clip(tmp, 0, 1)
    elif act1_type == 'HardSwish':
        act1_output = F.hardswish(input)

    else:
        raise TypeError(f"ConvActEleActFused not support activation \'{act1_type}\'")

    return act1_output


def Scale_Dot_Product_Attention_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Perform Scale Dot Product Attention opr forward.

    Args:
        op (Operation): SDP_Attention
        values (List[torch.Tensor]): opr inputs
        ctx (TorchBackendContext, optional): Context. Defaults to None.

    Returns:
        list: op output and internal result for quantization.
    """
    def get_parameters(op: Operation, attribute: str, default: Any = None):
        value = GET_ATTRIBUTE_FROM_OPERATION(op, attribute)
        if value is None or value == -1:
            value = default
        return value

    values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
    xq, xk, xv = values[:3]
    if len(xq.shape) == 4:
        # setup parameters
        B, N, num_heads, head_dim = xq.shape
        head_dim_ = get_parameters(op, attribute='head_dim', default=head_dim)
        assert head_dim == head_dim_

        scale = head_dim ** -0.5
        q = xq.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = xk.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = xv.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        energy = (q @ k.transpose(-2, -1)) * scale
        # Add atten_mask
        if len(values) > 3:
            energy = energy + values[3]

        attn = energy.softmax(dim=-1)

        feat = (attn @ v).transpose(1, 2)
        return feat
    
    elif  len(xq.shape) == 3:
        # setup parameters
        N, num_heads, head_dim = xq.shape
        head_dim_ = get_parameters(op, attribute='head_dim', default=head_dim)
        assert head_dim == head_dim_

        scale = head_dim ** -0.5
        q = xq.reshape(N, num_heads, head_dim).permute(1, 0, 2)
        k = xk.reshape(N, num_heads, head_dim).permute(1, 0, 2)
        v = xv.reshape(N, num_heads, head_dim).permute(1, 0, 2)

        energy = (q @ k.transpose(-2, -1)) * scale
        # Add atten_mask
        if len(values) > 3:
            energy = energy + values[3]

        attn = energy.softmax(dim=-1)

        feat = (attn @ v).transpose(1, 2)
        return feat

    else: 
        raise NotImplementedError('Not implement simplified MultiHeadAttention')



