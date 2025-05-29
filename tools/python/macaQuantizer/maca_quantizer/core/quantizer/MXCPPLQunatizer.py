
import torch

from typing import Union

from ppq.executor import BaseGraphExecutor
from ppq.quantization.optim import QuantizationOptimizationPipeline
from ppq.executor.op.torch.base import GET_ATTRIBUTE_FROM_OPERATION
from ppq.IR import BaseGraph
from ppq.api.setting import QuantizationSetting
from ppq.core import (PASSIVE_OPERATIONS,
                      OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)
from ppq.IR import BaseGraph, Operation

from ppq.quantization.quantizer.base import BaseQuantizer
from ppq.quantization.quantizer import PPLCUDAQuantizer

from maca_quantizer.core.config.config import FUSE_OPERATION_TYPE, PPLMETAX_ACTIVATION
from maca_quantizer.core.optim import MetaxConvElementwiseActivation


class MxcPPLQuantizer(PPLCUDAQuantizer):
    def __init__(self, graph: BaseGraph) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph)
        # self._num_of_bits = 8
        # self._quant_min = - int(pow(2, self._num_of_bits - 1) -1)
        # self._quant_max = int(pow(2, self._num_of_bits - 1) - 1)

    
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile')

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv', 'ConvTranspose'}:
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)
                conv_weight_config.observer_algorithm = 'minmax'
            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm'}:
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                gemm_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                gemm_weight_config.channel_axis = 0
                gemm_weight_config.observer_algorithm = 'minmax'
            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        # if operation.type in {'Div'}:
        #     if not operation.inputs[1].is_parameter:
        #         base_quant_config.input_quantization_config[1].state = QuantizationStates.FP32

        if operation.type in {'ConvActEleActFused'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            conv_weight_config = base_quant_config.input_quantization_config[1]
            conv_weight_config.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL +
                QuantizationProperty.LINEAR +
                QuantizationProperty.PER_CHANNEL
            )
            conv_weight_config.channel_axis = 0
            conv_weight_config.observer_algorithm = 'minmax'

            if bool(GET_ATTRIBUTE_FROM_OPERATION(operation, 'has_bias', 1)) and operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[2]
                bias_config.state = QuantizationStates.FP32

        elif operation.type in {'GroupNormalization', 'LayerNormalization','LayerNorm', 'Pow'}:
            for i in range(1, operation.num_of_input):
                if operation.inputs[i].is_parameter:
                    base_quant_config.input_quantization_config[i].state = QuantizationStates.FP32


        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        quant_pipeline = super().build_quant_pipeline(setting)
        # TODO: add for  ConvElementwiseActivation optimize
        if 'ConvActEleActFused' in FUSE_OPERATION_TYPE:
            quant_pipeline.append_optimization_to_pipeline(MetaxConvElementwiseActivation(), at_front=True)
        return quant_pipeline


    @ property
    def quant_operation_types(self) -> set:
        quant_ops =  {
            'Conv', 'Relu', 'PRelu', 'Clip', 'Gemm', 'MatMul',
            'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool', 'Squeeze',
            'Mul', 'Add', 'LeakyRelu', 'Split', 'Concat',
            'Transpose', 'Slice', 'Reshape', 'Flatten',
            'Sigmoid', 'ReduceMean', 'Softplus','Tanh',
            'Sub', 'Pow', 'ReduceSum', 'Sqrt','Div',
            'Gather', 'Pad', 'Softmax','LayerNormalization',
            'LayerNorm'
            # 'GroupNormalization',
            # 'LSTM',
            # 'InstanceNormalization',
            # 'ConvTranspose'

        }
        quant_ops = quant_ops | set(FUSE_OPERATION_TYPE) 
        # TODO: some operator int8 is not supported yet
        quant_ops.discard('MultiHeadAttentionV1')
        return quant_ops


    @ property
    def activation_fusion_types(self) -> set:
        fusion_ops = PPLMETAX_ACTIVATION
        # fusion_ops.discard('Swish')
        # fusion_ops.discard('HardSwish')
        return  fusion_ops  
