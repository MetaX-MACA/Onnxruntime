from enum import Enum

from maca_quantizer.core.parser import MACAOnnxParser, MACARUNTIMExporter
from maca_quantizer.core.quantizer import MxcPPLQuantizer, MxcTensorwiseQuantizer, MxcChannelwiseQuantizer
from maca_quantizer.core.operations import MXC_BACKEND_TABLE
from maca_quantizer.core.optim import (ComposeSwishPass, ComposeMishPass, ComposeHardSwishPass, ComposeHardSigmoidPass, 
                                       ComposeReduceL2Pass, ComposeConvElementwiseActivation, ComposeGroupNormPass, 
                                       ComposeSDPAttentionPass, ComposeGeluPass)

from maca_quantizer.ppq_.ppq import TargetPlatform, PPLCUDAQuantizer
from maca_quantizer.ppq_.ppq.parser import NativeExporter, PPLBackendExporter
from maca_quantizer.ppq_.ppq.api import (register_operation_handler, register_network_quantizer, 
                                         register_network_exporter, register_network_parser)



class RegisterFunction(Enum):
    PARSER    = register_network_parser,
    OPERATION = register_operation_handler,
    QUANRIZER = register_network_quantizer,
    EXPORTER  = register_network_exporter

# mxcppl register 
mxcppl_register = {
    'parser'    : MACAOnnxParser,
    'operation' : MXC_BACKEND_TABLE,
    'quantizer' : MxcPPLQuantizer,
    # 'quantizer' : MxcChannelwiseQuantizer,
    'exporter'  : None
}



# register table for difference ep
REGISTER_TABLE = {
    'mxc_pplcuda'   : mxcppl_register
}

# quantize method 
MXPPL_ALGORITHM_PLATFORM ={
    "pertensor":  TargetPlatform.PPL_CUDA_INT8,
    "perchannel": TargetPlatform.PPL_CUDA_INT8
}

# export model tyepe
class EXPORTER_INSTANCE(Enum):
    native = NativeExporter
    onnx   = MACARUNTIMExporter
    ppl    = PPLBackendExporter



# fuse operation table
FUSE_OPERATION_TABLE = {
    'Swish'       : ComposeSwishPass, 
    'Mish'        : ComposeMishPass,  
    'HardSwish'   : ComposeHardSwishPass, 
    'HardSigmoid' : ComposeHardSigmoidPass,
    'ReduceL2'    : ComposeReduceL2Pass,
    'Gelu'        : ComposeGeluPass,
    'GroupNormalization' : ComposeGroupNormPass,
    # 'ConvActEleActFused': ComposeConvElementwiseActivation
    'MultiHeadAttentionV1': ComposeSDPAttentionPass
}






if __name__=="__main__":
    print(RegisterFunction.OPERATION.name)
    print(type(RegisterFunction.OPERATION.name))