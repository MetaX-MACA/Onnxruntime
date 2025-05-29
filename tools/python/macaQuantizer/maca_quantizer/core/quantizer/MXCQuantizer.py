from typing import Union

import torch
from ppq.api.setting import QuantizationSetting
from ppq.core import (PASSIVE_OPERATIONS,
                      OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)
from ppq.executor.base import BaseGraphExecutor
from ppq.IR import BaseGraph, Operation
from ppq.quantization.optim.base import QuantizationOptimizationPipeline
from ppq.quantization.optim.morph import MetaxGemmSplitPass

from ppq.quantization.quantizer.base import BaseQuantizer
from maca_quantizer.core.config.config import FUSE_OPERATION_TYPE


class MxcTensorwiseQuantizer(BaseQuantizer):
    def __init__(
        self,
        graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = -128
        self._quant_max = 127

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
 
        base_quant_config = self.create_default_quant_config(
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile', policy=self.quantize_policy,
            rounding=self.rounding_policy,
        )

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm'}:
            weight_config = base_quant_config.input_quantization_config[1]
            weight_config.quant_min = -127
            weight_config.quant_max = 127

            # if operation has bias, bias should be quantized with fp32/int32
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                # bias_config.state = QuantizationStates.FP32
                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_TENSOR
                )
                bias_config.num_of_bits = 32
                bias_config.quant_max = int(pow(2, bias_config.num_of_bits - 1) - 1)
                bias_config.quant_min = - int(pow(2, bias_config.num_of_bits - 1))
                bias_config.state = QuantizationStates.PASSIVE_INIT

            for config in base_quant_config.input_quantization_config[1: ]:
                config.observer_algorithm = 'minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.METAX_INT8_T

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'Relu', 'PRelu', 'Clip', 'Gemm',
            'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'LeakyRelu', 'Split', 'Concat',
            'Transpose', 'Slice', 'Reshape', 'Flatten',
            'MatMul'}

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR)

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    def build_prequant_pipeline(
        self, setting: QuantizationSetting, executor: BaseGraphExecutor) -> QuantizationOptimizationPipeline:
        return super().build_prequant_pipeline(setting, executor)

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}


# TODO: SET PERCHANNEL QUANTIZER
class MxcChannelwiseQuantizer(BaseQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, BaseGraph]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = 0
        self._quant_max = 255

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in  {'Conv', 'ConvTranspose', 'Gemm'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # Conv layout:           [out_channel, in_channel,  kernel_size, kernel_size]
            # ConvTranspose layout:  [in_channel,  out_channel, kernel_size, kernel_size]
            conv_weight_config = base_quant_config.input_quantization_config[1]
            conv_weight_config.quant_max = 127
            conv_weight_config.quant_min = -127
            conv_weight_config.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL +
                QuantizationProperty.LINEAR +
                QuantizationProperty.PER_CHANNEL
            )
            c_index = 1 if operation.type in {'ConvTranspose'} else 0

            conv_weight_config.channel_axis = c_index
            conv_weight_config.observer_algorithm = 'Minmax'

            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32
                # bias_config.policy = QuantizationPolicy(
                #     QuantizationProperty.SYMMETRICAL +
                #     QuantizationProperty.LINEAR +
                #     QuantizationProperty.PER_CHANNEL
                # )
                # bias_config.num_of_bits = 32
                # bias_config.quant_max = int(pow(2, bias_config.num_of_bits - 1) - 1)
                # bias_config.quant_min = - int(pow(2, bias_config.num_of_bits - 1))
                # bias_config.state = QuantizationStates.PASSIVE_INIT
                # bias_config.channel_axis = 0
                # bias_config.observer_algorithm = 'minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.METAX_INT8_C

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        quant_ops =  {
            'Conv', 'Relu', 'PRelu', 'Clip', 'Gemm',
            'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'LeakyRelu', 'Split', 'Concat',
            'Transpose', 'Slice', 'Reshape', 'Flatten',
            'Sigmoid', 'ReduceMean', 'Softplus','Tanh',
            'Sub', 'Pow', 'ReduceSum', 'Sqrt','Div',
            'Gather', 'Pad',
        }
        quant_ops = quant_ops | set(FUSE_OPERATION_TYPE) 
        return quant_ops


    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.ASYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    def build_prequant_pipeline(
        self, setting: QuantizationSetting, executor: BaseGraphExecutor) -> QuantizationOptimizationPipeline:
        return super().build_prequant_pipeline(setting, executor)

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}
