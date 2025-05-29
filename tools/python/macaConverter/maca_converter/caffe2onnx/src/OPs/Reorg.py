import onnx
from typing import *
from onnx import helper
from typing import *
import ctypes
import caffe2onnx.src.c2oObject as Node
import numpy as np

def create_attributes(layer) -> Dict:
    #stride = layer.reorg_param.stride

    attributes = {
        "blocksize": 2
    }

    return attributes

def caculate_output_shape(layer, input_shape) -> List:
    #stride = 2
    output_shape = [input_shape[0][0], input_shape[0][1]*4, int(input_shape[0][2]/2), int(input_shape[0][3]/2)]
    print('reorg output_shape:', output_shape)
    return [output_shape]

def create_reorg_node(layer,
                       node_name: str,
                       inputs_name: List[str],
                       outputs_name: List[str],
                       inputs_shape: List, ) -> onnx.NodeProto:
    attributes = create_attributes(layer)

    outputs_shape = caculate_output_shape(layer, inputs_shape)

    node = Node.c2oNode(layer, node_name, "SpaceToDepth",
                        inputs_name, outputs_name,
                        inputs_shape, outputs_shape,
                        attributes)
    return node

