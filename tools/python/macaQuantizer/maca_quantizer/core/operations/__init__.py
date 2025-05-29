
from .customize_ops import *
from .official_ops import *


MXC_BACKEND_TABLE = {
    'ArgMax'          : ArgMax_forward,
    'ArgMin'          : ArgMin_forward,
    'AveragePool'     : AveragePool_forward,
    'Ceil'            : Ceil_forward, 
    'DepthToSpace'    : DepthToSpace_forward,
    'MaxUnpool'       : MaxUnpool_forward,
    'Mish'            : Mish_forward,
    'Reshape'         : Reshape_forward,
    'Resize'          : Resize_forward,
    'ReduceMin'       : ReduceMin_forward,
    'ReduceProd'      : ReduceProd_forward,
    'Round'           : Round_forward,
    'Swish'           : Swish_forward,
    'ScatterElements' : ScatterElements_forward,
    'SequenceAt'      : SequenceAt_forward,
    'SplitToSequence' : SplitToSequence_forward,
    'CumSum'          : CumSum_forward,
    'Einsum'          : Einsum_forward,
    'Xor'             : Xor_forward,
    'LSTM'            : LSTM_forward,
    'GroupNormalization': GroupNormalization_forward,

    'ConvActEleActFused': ConvActEleActFused_forward,
    'MultiHeadAttentionV1': Scale_Dot_Product_Attention_forward
    
}