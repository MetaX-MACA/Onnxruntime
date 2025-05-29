import onnx
from onnx import numpy_helper, helper
import numpy as np
import operation


def get_gemm_bn(model):
    gemm_bn_list = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            bn_node, ok = operation.get_next_node_by_output(model, node.output[0])
            if ok == 0 and bn_node.op_type == 'BatchNormalization':
                node_dict = {}
                node_dict['Gemm'] = node
                node_dict['BatchNormalization'] = bn_node
                gemm_bn_list.append(node_dict)

    return gemm_bn_list


def handle_gemm_bn(model, gemm_bn_dict):
    gemm_node = gemm_bn_dict['Gemm']
    bn_node = gemm_bn_dict['BatchNormalization']

    for attr in gemm_node.attribute:
        if attr.name == 'transB':
            gemm_transB = attr.i
        if attr.name == 'beta':
            gemm_beta = attr.f
    gemm_b = next(initializer for initializer in model.graph.initializer if initializer.name == gemm_node.input[1])
    gemm_c = next((initializer for initializer in model.graph.initializer if initializer.name == gemm_node.input[2]), None)
    gemm_b = numpy_helper.to_array(gemm_b)
    gemm_c = numpy_helper.to_array(gemm_c) if gemm_c is not None else np.array([0]).astype(gemm_b.dtype)
    if gemm_transB != 0:
        gemm_b = np.transpose(gemm_b)

    bn_epsilon = 0.00001
    for attr in bn_node.attribute:
        if attr.name == 'epsilon':
            bn_epsilon = attr.f
            break
    bn_scale = next(initializer for initializer in model.graph.initializer if initializer.name == bn_node.input[1])
    bn_bias = next(initializer for initializer in model.graph.initializer if initializer.name == bn_node.input[2])
    bn_mean = next(initializer for initializer in model.graph.initializer if initializer.name == bn_node.input[3])
    bn_var = next(initializer for initializer in model.graph.initializer if initializer.name == bn_node.input[4])
    bn_var = numpy_helper.to_array(bn_var)
    bn_mean = numpy_helper.to_array(bn_mean)
    bn_bias = numpy_helper.to_array(bn_bias)
    bn_scale = numpy_helper.to_array(bn_scale)

    sq = np.sqrt(bn_var + bn_epsilon)
    fused_gemm_c = (gemm_c * bn_scale - (bn_mean * bn_scale - bn_bias * sq) / gemm_beta) / sq
    fused_gemm_c = fused_gemm_c.squeeze()

    fused_type = onnx.TensorProto.FLOAT
    if fused_gemm_c.dtype == np.float16:
        fused_type = onnx.TensorProto.FLOAT16
    gemm_c_name = (gemm_node.input[2] if len(gemm_node.input) > 2 else gemm_node.name) + '_fuse_bn'
    gemm_c = helper.make_tensor(gemm_c_name, fused_type, fused_gemm_c.shape, fused_gemm_c.flatten())
    model.graph.initializer.extend([gemm_c])

    fused_gemm_b = np.transpose(gemm_b * bn_scale / sq)
    gemm_b_name = gemm_node.input[1] + '_fuse_bn'
    gemm_b = helper.make_tensor(gemm_b_name, fused_type, fused_gemm_b.shape, fused_gemm_b.flatten())
    model.graph.initializer.extend([gemm_b])

    gemm_node.name = gemm_node.name + '+' + bn_node.name
    gemm_node.input[1] = gemm_b_name
    gemm_node.input[2] = gemm_c_name
    gemm_node.output[0] = bn_node.output[0]
    if len(bn_node.output) > 1:
        for i in range(1, len(bn_node.output)):
            gemm_node.output.append(bn_node.output[i])

    operation.remove_onnx_node(model, bn_node)
    operation.remove_unused_initializer(model)


def gemm_bn_fuse(model):
    gemm_bn_list = get_gemm_bn(model)

    for gemm_bn_dict in gemm_bn_list:
        handle_gemm_bn(model, gemm_bn_dict)

    return model
