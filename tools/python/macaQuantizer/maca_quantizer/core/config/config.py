# global constant parameters for maca_quantizer


# 图结构可以等价融合的算子 Operators that can be equivalently fused
FUSE_OPERATION_TYPE = {'Swish', 'Mish', 'HardSwish', 'HardSigmoid', 'ReduceL2', 'Gelu', 'ConvActEleActFused', 'GroupNormalization', 'MultiHeadAttentionV1'}

# PPL METAX 中所有需要与 Conv 融合的激活函数
# [Conv -- Act -- Elementwise -- Act] fusion support activation type
PPLMETAX_ACTIVATION = {'Relu', 'Sigmoid', 'Swish', 'HardSwish', 'HardSigmoid', 'Clip', 'LeakyRelu', 'PRelu', 'Mish'}

ACTIVATIONS_TYPE = {'Relu', 'Clip', 'PRelu','LeakyRelu','Sigmoid', 'Mish', 'Swish', 'HardSwish', 'HardSigmoid', 'Softplus', 'Softmax'}

QUANT_COMPUTING_OP = {'Conv', 'Gemm', 'MatMul', 'Attention', 'ConvActEleActFused'}


MXQ_USE_MACART_BACKEND=1
DISPATCH_FP32_ATTRIB_KEY = 'dispatch_fp32'

# analyse error threshold
SNR_ERROR_THRESHOLD = 0.1
COSIN_ERROR_THRESHOLD = 0.95
